"""Dataset download, conversion, de-duplication and splitting.

Output: one YOLO dataset with classes 0 = fire, 1 = smoke:
    <out>/images/{train,val,test}/*.jpg   <out>/labels/{train,val,test}/*.txt   <out>/data.yaml
Every source is converted to the same class ids, images are resized (long side <= max_side),
exact and near duplicates are removed (perceptual hash), and near-duplicate / consecutive-frame
groups are kept inside one split so validation is not inflated by leakage.
"""
import base64
import csv
import json
import os
import random
import re
import shutil
import tarfile
import time
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from .common import CLASS_NAMES, IMG_EXTS, LABEL_ALIASES, VID_EXTS, fmt_secs, list_files, save_yaml

DFIRE_ID_MAP = {0: "smoke", 1: "fire"}   # D-Fire: 0 = smoke, 1 = fire (opposite of ours!)
FASDD_ID_MAP = {0: "fire", 1: "smoke"}   # FASDD: 0 = fire, 1 = smoke


# --------------------------------------------------------------------------------------- downloads
def _retry(fn, tries=6, base=5, what=""):
    for i in range(tries):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            if i == tries - 1:
                raise
            wait = base * 2 ** i
            print(f"   {what}: attempt {i + 1}/{tries} failed ({type(e).__name__}: {str(e)[:120]}) - retry in {wait}s")
            time.sleep(wait)


def hf_list(repo_id):
    from huggingface_hub import HfApi
    return _retry(lambda: HfApi().list_repo_files(repo_id, repo_type="dataset"), what=f"list {repo_id}")


def hf_get(repo_id, filename, dest_dir):
    from huggingface_hub import hf_hub_download
    return _retry(lambda: hf_hub_download(repo_id, filename, repo_type="dataset", local_dir=str(dest_dir)),
                  what=f"{repo_id}/{filename}")


def url_get(url, dest):
    import requests
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    def _go():
        with requests.get(url, stream=True, timeout=60, allow_redirects=True) as r:
            r.raise_for_status()
            tmp = dest.with_suffix(dest.suffix + ".part")
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(1 << 22):
                    f.write(chunk)
            tmp.rename(dest)
        return dest
    return _retry(_go, tries=3, what=url[:60])


def onedrive_direct(share_url):
    """Public OneDrive share link -> direct download URL."""
    b64 = base64.urlsafe_b64encode(share_url.encode()).decode().rstrip("=")
    return f"https://api.onedrive.com/v1.0/shares/u!{b64}/root/content"


def extract_archive(path, out_dir):
    """Extract zip/tar once (a marker file makes re-runs instant)."""
    path, out_dir = Path(path), Path(out_dir)
    marker = out_dir / f".extracted_{path.name}"
    if marker.exists():
        return out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"   extracting {path.name} ({path.stat().st_size / 1e9:.2f} GB) -> {out_dir}")
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z:
            z.extractall(out_dir)
    elif tarfile.is_tarfile(path):
        with tarfile.open(path) as t:
            t.extractall(out_dir)
    else:
        raise ValueError(f"not a zip/tar archive: {path}")
    marker.touch()
    return out_dir


def _manual_drop(raw_dir, name, extract_dir):
    """If the user put a zip/tar or an unpacked folder in raw/<name>/, use it."""
    d = Path(raw_dir) / name
    if not d.exists():
        return None
    archives = [p for p in d.iterdir() if p.is_file() and (p.suffix.lower() in {".zip", ".tar", ".gz", ".tgz"})]
    for a in archives:
        extract_archive(a, Path(extract_dir) / name)
    if archives:
        return Path(extract_dir) / name
    if list_files(d, IMG_EXTS)[:1]:
        return d
    return None


def _parquet_to_yolo(parquet_files, out_dir):
    """HF parquet export (image bytes + YOLO label text) -> images/ + labels/ folders."""
    import pyarrow.parquet as pq
    out_dir = Path(out_dir)
    if (out_dir / ".done").exists():
        return out_dir
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "labels").mkdir(parents=True, exist_ok=True)
    n = 0
    for pf in parquet_files:
        f = pq.ParquetFile(pf)
        cols = f.schema_arrow.names
        img_col = next((c for c in cols if "image" in c.lower()), None)
        lab_col = next((c for c in cols if c.lower() in {"label", "labels", "annotation", "annotations", "txt"}), None)
        fn_col = next((c for c in cols if c.lower() in {"filename", "file_name", "name", "path", "image_id"}), None)
        if img_col is None or lab_col is None:
            raise ValueError(f"unexpected parquet columns {cols}")
        for batch in f.iter_batches(batch_size=256):
            d = batch.to_pydict()
            for i, img in enumerate(d[img_col]):
                raw = img.get("bytes") if isinstance(img, dict) else img
                if not raw:
                    continue
                lab = d[lab_col][i]
                if lab is not None and not isinstance(lab, str):
                    raise ValueError("label column is not YOLO text")
                fn = d[fn_col][i] if fn_col else (img.get("path") if isinstance(img, dict) else None)
                stem = Path(str(fn)).stem if fn else f"img_{n:07d}"
                (out_dir / "images" / f"{stem}.jpg").write_bytes(raw)
                (out_dir / "labels" / f"{stem}.txt").write_text(lab or "")
                n += 1
    (out_dir / ".done").touch()
    print(f"   parquet -> {n:,} images")
    return out_dir


DFIRE_ONEDRIVE = "https://1drv.ms/u/c/c0bd25b6b048b01d/EbLgD7bES4FDvUN37Grxn8QBF5gIBBc7YV2qklF08GCiBw"
DFIRE_HF_PARQUET = ["badsaarow/d-fire"]                 # Hugging Face mirrors, tried in this order; each is CHECKED after download
DFIRE_HF_ZIP = ["Stevebirth/fire-ulm-dfire"]
FASDD_HF_REPOS = ["weslleyskah/fire_smoke_dataset_fasdd_cv", "seawsurf/fire_smoke_dataset_fasdd_cv"]
SENTINEL_REPO = "kalyan1729/sentineledgedataset"         # Bengaluru CCTV + UA-DETRAC traffic frames (CC BY 4.0): fire-free negatives
FIRESENSE_URLS = {"fire": "https://zenodo.org/records/836749/files/fire_videos.1406.zip?download=1",
                  "smoke": "https://zenodo.org/records/836749/files/smoke_videos.1407.zip?download=1"}


def _nums(vals):
    try:
        for v in vals:
            float(v)
        return True
    except ValueError:
        return False


def yolo_folder_ok(root, min_pos_frac=0.2, sample=400, seed=0):
    """Sanity check of a downloaded folder: do its .txt files hold valid 'class cx cy w h' lines?
    Guards against mirrors whose label column is something else (then the next source is tried)."""
    lbls = [p for p in Path(root).rglob("*.txt") if p.stem.lower() not in SKIP_TXT]
    if len(lbls) < 50:
        return False
    random.Random(seed).shuffle(lbls)
    lbls = lbls[:sample]
    pos = bad = 0
    for p in lbls:
        try:
            rows = [l.split() for l in p.read_text(errors="ignore").splitlines() if l.strip()]
        except Exception:  # noqa: BLE001
            bad += 1
            continue
        if not rows:
            continue
        good = all(len(r) >= 5 and _nums(r[:5]) and 0 <= float(r[1]) <= 1 and 0 <= float(r[2]) <= 1
                   and 0 < float(r[3]) <= 1 and 0 < float(r[4]) <= 1 for r in rows)
        pos += int(good)
        bad += int(not good)
    return bad <= 0.05 * len(lbls) and pos >= min_pos_frac * len(lbls)


def get_dfire(raw_dir, extract_dir):
    """Returns a folder containing D-Fire images + YOLO labels (0=smoke, 1=fire), or None."""
    raw_dir, extract_dir = Path(raw_dir), Path(extract_dir)
    p = _manual_drop(raw_dir, "dfire", extract_dir)
    if p:
        print("D-Fire: using files you placed in raw/dfire/")
        return p
    for repo in DFIRE_HF_PARQUET:          # 1) Hugging Face parquet mirror
        try:
            files = [f for f in hf_list(repo) if f.endswith(".parquet")]
            if not files:
                raise RuntimeError("no parquet files")
            print(f"D-Fire: downloading {len(files)} parquet files from HF {repo}")
            local = [hf_get(repo, f, raw_dir / "dfire_hf_parquet") for f in files]
            out = _parquet_to_yolo(local, extract_dir / "dfire_parquet")
            if yolo_folder_ok(out):
                return out
            print(f"D-Fire: {repo} does not contain usable YOLO labels - trying the next source")
        except Exception as e:  # noqa: BLE001
            print(f"D-Fire: {repo} failed ({str(e)[:150]})")
    for repo in DFIRE_HF_ZIP:              # 2) Hugging Face zip mirror
        try:
            zips = [f for f in hf_list(repo) if f.lower().endswith(".zip")]
            if not zips:
                raise RuntimeError("no zip")
            print(f"D-Fire: downloading {zips} from HF {repo}")
            for z in zips:
                extract_archive(hf_get(repo, z, raw_dir / "dfire_hf_zip"), extract_dir / "dfire_zip")
            if yolo_folder_ok(extract_dir / "dfire_zip"):
                return extract_dir / "dfire_zip"
            print(f"D-Fire: {repo} does not contain usable YOLO labels - trying the next source")
        except Exception as e:  # noqa: BLE001
            print(f"D-Fire: {repo} failed ({str(e)[:150]})")
    try:                                   # 3) official OneDrive link
        dest = raw_dir / "dfire_onedrive" / "D-Fire.zip"
        if not dest.exists():
            print("D-Fire: downloading from the official OneDrive link")
            url_get(onedrive_direct(DFIRE_ONEDRIVE), dest)
        out = extract_archive(dest, extract_dir / "dfire_onedrive")
        if yolo_folder_ok(out):
            return out
        print("D-Fire: the OneDrive archive has no usable YOLO labels")
    except Exception as e:  # noqa: BLE001
        print(f"D-Fire: OneDrive failed ({str(e)[:150]})")
    print("D-Fire: ALL automatic downloads failed. Download it yourself from "
          "https://github.com/gaia-solutions-on-demand/DFireDataset and put the zip in raw/dfire/, then re-run.")
    return None


def get_fasdd(raw_dir, extract_dir):
    raw_dir, extract_dir = Path(raw_dir), Path(extract_dir)
    p = _manual_drop(raw_dir, "fasdd", extract_dir)
    if p:
        print("FASDD: using files you placed in raw/fasdd/")
        return p
    for repo in FASDD_HF_REPOS:
        owner = repo.split("/")[0]
        try:
            files = hf_list(repo)
            arch = [f for f in files if f.lower().endswith((".zip", ".tar", ".tar.gz", ".tgz"))]
            parq = [f for f in files if f.endswith(".parquet")]
            if arch:
                print(f"FASDD: downloading {arch} from HF {repo}")
                for z in arch:
                    extract_archive(hf_get(repo, z, raw_dir / "fasdd_hf" / owner), extract_dir / "fasdd" / owner)
                root = extract_dir / "fasdd" / owner
            elif parq:
                print(f"FASDD: downloading {len(parq)} parquet files from HF {repo}")
                local = [hf_get(repo, f, raw_dir / "fasdd_hf" / owner) for f in parq]
                root = _parquet_to_yolo(local, extract_dir / "fasdd" / owner)
            else:
                from huggingface_hub import snapshot_download
                print(f"FASDD: downloading {len(files)} files from HF {repo} (a few GB)")
                root = Path(_retry(lambda: snapshot_download(repo, repo_type="dataset", local_dir=str(raw_dir / "fasdd_hf" / owner),
                                                              max_workers=16), what=repo))
            if yolo_folder_ok(root) or list(Path(root).rglob("*.xml"))[:1]:
                return root
            print(f"FASDD: {repo} does not contain usable labels - trying the next source")
        except Exception as e:  # noqa: BLE001
            print(f"FASDD: {repo} failed ({str(e)[:150]})")
    print("FASDD: not available. Optional - download FASDD_CV from Science Data Bank "
          "(https://doi.org/10.57760/sciencedb.j00104.00103) and put the zip in raw/fasdd/.")
    return None


def _hf_download_retry(repo, filename, local_dir, tries=3):
    from huggingface_hub import hf_hub_download
    for i in range(tries):
        try:
            return hf_hub_download(repo, filename, repo_type="dataset", local_dir=str(local_dir))
        except Exception:  # noqa: BLE001
            if i == tries - 1:
                return None
            time.sleep(2 * 2 ** i)


def get_street_frames(raw_dir, n_train=2000, n_mine=800, n_test=400, seed=0, workers=16, max_side=960):
    """Fire-free CCTV traffic frames from the SentinelEdge dataset (Hugging Face, CC BY 4.0).
      train: Bengaluru CCTV (BMD-45) + UA-DETRAC  -> hard negatives for training
      mine : Bengaluru UVH-26 folders 000-001     -> the detector's false alarms on them are mined (detector v2)
      test : UVH-26 folder 002 + IITM-HeTra       -> other cameras, only used to measure the false-alarm rate
    Frames are saved as JPEGs with the long side <= max_side. Returns {bucket: folder} for the buckets that are available."""
    import cv2
    raw_dir = Path(raw_dir)
    base = raw_dir / "street_frames"
    plan = {"train": ([("BMD-45-Train/", 0.65), ("DETRAC/DETRAC-Images/", 0.35)], n_train),
            "mine": ([("UVH-26-Train/data/000/", 0.5), ("UVH-26-Train/data/001/", 0.5)], n_mine),
            "test": ([("UVH-26-Train/data/002/", 0.6), ("IITM-HeTra_v2/", 0.4)], n_test)}

    def available():
        return {k: base / k for k in plan if (base / k / ".done").exists() and len(list_files(base / k, IMG_EXTS)) >= 20}

    if len(available()) == len(plan) or not any(n for _, n in plan.values()):
        return available()
    try:
        files = hf_list(SENTINEL_REPO)
    except Exception as e:  # noqa: BLE001
        print(f"street frames: cannot list {SENTINEL_REPO} ({str(e)[:120]}) - skipped (the notebook still runs without them)")
        return available()
    rng = random.Random(seed)
    img_ext = {".png", ".jpg", ".jpeg"}
    jobs = []
    for bucket, (parts, n) in plan.items():
        if (base / bucket / ".done").exists():
            continue
        for prefix, frac in parts:
            cand = [f for f in files if f.startswith(prefix) and Path(f).suffix.lower() in img_ext]
            if not cand:
                print(f"street frames: nothing found under {prefix} - that part is skipped")
                continue
            rng.shuffle(cand)
            jobs += [(bucket, f) for f in cand[:int(n * frac)]]
    print(f"street frames: downloading {len(jobs):,} frames from {SENTINEL_REPO} ...")
    tmp = base / "_tmp"
    stats = Counter()
    t0 = time.time()

    def work(job):
        bucket, f = job
        dest = base / bucket / f"{Path(f).parent.name}__{Path(f).stem}.jpg"
        if dest.exists():
            return bucket, True
        p = _hf_download_retry(SENTINEL_REPO, f, tmp)
        if p is None:
            return bucket, False
        try:
            img = cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                return bucket, False
            dest.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(dest), _resize(img, max_side), [cv2.IMWRITE_JPEG_QUALITY, 92])
            return bucket, True
        except Exception:  # noqa: BLE001
            return bucket, False
        finally:
            try:
                os.remove(p)
            except OSError:
                pass

    done = 0
    with ThreadPoolExecutor(workers) as ex:
        for bucket, ok in ex.map(work, jobs):
            stats[(bucket, ok)] += 1
            done += 1
            if done % 500 == 0 or done == len(jobs):
                el = time.time() - t0
                print(f"   street frames: {done:,}/{len(jobs):,} elapsed {fmt_secs(el)} ETA {fmt_secs(el / done * (len(jobs) - done))}")
    shutil.rmtree(tmp, ignore_errors=True)
    for bucket in plan:
        ok_n, bad_n = stats[(bucket, True)], stats[(bucket, False)]
        if ok_n >= 20 and ok_n >= 0.7 * (ok_n + bad_n):
            (base / bucket).mkdir(parents=True, exist_ok=True)
            (base / bucket / ".done").touch()
        if ok_n or bad_n:
            print(f"   {bucket}: {ok_n:,} frames ok, {bad_n:,} failed")
    return available()


def _firesense_label(rel_parts):
    """FIRESENSE file/folder names -> 'event' (positive), 'normal' (negative) or None (cannot tell)."""
    s = "/".join(rel_parts).lower()
    toks = {t for t in re.split(r"[^a-z]+", s) if t}
    if (toks & {"neg", "negative", "negatives", "normal", "nonfire", "nofire", "nosmoke", "nonsmoke", "false"}
            or "non" in toks or ("no" in toks and toks & {"fire", "smoke", "flame"})):
        return "normal"
    if toks & {"pos", "positive", "positives"}:
        return "event"
    return None


def get_firesense(raw_dir, eval_dir):
    """FIRESENSE test videos (Zenodo 836749, CC BY 4.0) -> eval_dir/{fire,smoke,normal}/.
    Positive / negative videos are told apart by their file and folder names; if that is not possible the
    videos are skipped (and you can sort them by hand). Returns the number of videos copied per folder."""
    raw_dir, eval_dir = Path(raw_dir), Path(eval_dir)
    marker = eval_dir / ".firesense_done"
    if marker.exists():
        return json.loads(marker.read_text())
    counts = {"fire": 0, "smoke": 0, "normal": 0}
    for kind, url in FIRESENSE_URLS.items():
        try:
            dest = raw_dir / "firesense" / f"{kind}_videos.zip"
            if not dest.exists():
                print(f"FIRESENSE: downloading the {kind} videos from Zenodo ...")
                url_get(url, dest)
            root = extract_archive(dest, raw_dir / "firesense" / f"{kind}_extracted")
        except Exception as e:  # noqa: BLE001
            print(f"FIRESENSE {kind}: download failed ({str(e)[:150]}) - skipped, it is retried on the next run")
            return counts
        vids = list_files(root, VID_EXTS)
        lab = [(v, _firesense_label(v.relative_to(root).parts)) for v in vids]
        pos = [v for v, l in lab if l == "event"]
        neg = [v for v, l in lab if l == "normal"]
        unk = [v for v, l in lab if l is None]
        print(f"FIRESENSE {kind}: {len(vids)} videos -> {len(pos)} with {kind}, {len(neg)} without, {len(unk)} not recognised "
              f"(e.g. {[str(v.relative_to(root)) for v in vids[:3]]})")
        if len(pos) < 3 or len(neg) < 3 or len(unk) > 0.2 * max(1, len(vids)):
            print(f"   cannot tell the positive from the negative {kind} videos by name -> not used. "
                  f"Sort them by hand into raw/eval_videos/{kind}/ and raw/eval_videos/normal/ (they are in {root}).")
            continue
        for sub in (kind, "normal"):
            (eval_dir / sub).mkdir(parents=True, exist_ok=True)
        for v in pos:
            shutil.copy(v, eval_dir / kind / f"fs_{kind}_{v.name}")
            counts[kind] += 1
        for v in neg:
            shutil.copy(v, eval_dir / "normal" / f"fs_{kind}_{v.name}")
            counts["normal"] += 1
    if sum(counts.values()):
        marker.write_text(json.dumps(counts))
    print("FIRESENSE test videos:", counts)
    return counts


# --------------------------------------------------------------------------------------- ingestion
def _names_from_folder(root):
    """Find class names for YOLO txt labels (data.yaml / classes.txt / obj.names)."""
    import yaml
    for p in list(Path(root).rglob("data.yaml")) + list(Path(root).rglob("*.yaml")):
        try:
            d = yaml.safe_load(p.read_text())
            names = d.get("names") if isinstance(d, dict) else None
            if isinstance(names, dict):
                return {int(k): str(v) for k, v in names.items()}
            if isinstance(names, list):
                return dict(enumerate(map(str, names)))
        except Exception:  # noqa: BLE001
            continue
    for nm in ("classes.txt", "obj.names", "_darknet.labels"):
        for p in Path(root).rglob(nm):
            lines = [x.strip() for x in p.read_text().splitlines() if x.strip()]
            if lines:
                return dict(enumerate(lines))
    return None


def _id_map_for(root, default, name):
    """Class ids of a downloaded dataset: read from its own data.yaml / classes file when it names fire AND smoke,
    otherwise the documented default for that dataset."""
    names = _names_from_folder(root)
    if names and {_canon(v) for v in names.values()} >= {"fire", "smoke"}:
        print(f"[{name}] class names from dataset files: {names}")
        return names
    return default


def _canon(name):
    return LABEL_ALIASES.get(str(name).strip().lower().replace(" ", "_").replace("-", "_"))


def _parse_voc(xml_path):
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    wh = None
    if size is not None:
        try:
            wh = (float(size.findtext("width")), float(size.findtext("height")))
            if wh[0] <= 1 or wh[1] <= 1:
                wh = None
        except (TypeError, ValueError):
            wh = None
    objs = []
    for o in root.iter("object"):
        c = _canon(o.findtext("name", ""))
        bb = o.find("bndbox")
        if c is None or bb is None:
            continue
        try:
            objs.append((c, float(bb.findtext("xmin")), float(bb.findtext("ymin")),
                         float(bb.findtext("xmax")), float(bb.findtext("ymax"))))
        except (TypeError, ValueError):
            continue
    return objs, wh


def _parse_yolo(txt_path, id_map):
    objs = []
    for line in Path(txt_path).read_text(errors="ignore").splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        try:
            cid, cx, cy, w, h = int(float(p[0])), float(p[1]), float(p[2]), float(p[3]), float(p[4])
        except ValueError:
            continue
        c = _canon(id_map.get(cid, "")) if id_map else None
        if c is None:
            continue
        objs.append((c, cx, cy, w, h))
    return objs


SKIP_TXT = {"train", "val", "valid", "test", "classes", "obj", "readme", "license", "labels", "notes"}


def ingest_folder(name, root, id_map=None, cap=None, max_neg_frac=None, seed=0, negatives_only=False,
                  force_split=None, group_fn=None):
    """Turn any image folder (YOLO txt or VOC xml labels) into records.

    record = dict(src, path, fmt('yolo'|'voc'|'neg'), objs, wh, group, force_split)
    """
    root = Path(root)
    imgs = list_files(root, IMG_EXTS)
    if not imgs:
        print(f"[{name}] no images found under {root}")
        return []
    recs = []
    if negatives_only:
        for p in imgs:
            recs.append(dict(src=name, path=str(p), fmt="neg", objs=[], wh=None,
                             group=group_fn(p) if group_fn else None, force_split=force_split))
        print(f"[{name}] {len(recs):,} negative images")
        return _cap(recs, cap, max_neg_frac, seed, name)

    txt_index, xml_index = defaultdict(list), defaultdict(list)
    for p in root.rglob("*.txt"):
        if p.stem.lower() not in SKIP_TXT:
            txt_index[p.stem].append(p)
    for p in root.rglob("*.xml"):
        xml_index[p.stem].append(p)
    if id_map is None and txt_index:
        names = _names_from_folder(root)
        if names:
            id_map = names
            print(f"[{name}] class names from dataset files: {names}")
    missing, bad = 0, 0
    for p in imgs:
        rec = dict(src=name, path=str(p), fmt=None, objs=[], wh=None,
                   group=group_fn(p) if group_fn else None, force_split=force_split)
        # 1) YOLO sibling .../images/x.jpg -> .../labels/x.txt
        parts = list(p.parts)
        sib = None
        if "images" in parts:
            i = len(parts) - 1 - parts[::-1].index("images")
            sib = Path(*parts[:i], "labels", *parts[i + 1:]).with_suffix(".txt")
        try:
            if sib is not None and sib.exists() and id_map:
                rec.update(fmt="yolo", objs=_parse_yolo(sib, id_map))
            elif len(xml_index.get(p.stem, [])) >= 1:
                objs, wh = _parse_voc(xml_index[p.stem][0])
                rec.update(fmt="voc", objs=objs, wh=wh)
            elif len(txt_index.get(p.stem, [])) >= 1 and id_map:
                cands = sorted(txt_index[p.stem], key=lambda q: ("yolo" not in str(q).lower(), len(str(q))))
                rec.update(fmt="yolo", objs=_parse_yolo(cands[0], id_map))
            else:
                missing += 1
                continue
        except Exception:  # noqa: BLE001
            bad += 1
            continue
        recs.append(rec)
    if txt_index and not id_map and not xml_index:
        raise ValueError(f"[{name}] YOLO .txt labels found but no class-name mapping. Pass id_map, e.g. {{0: 'fire', 1: 'smoke'}}")
    n_pos = sum(1 for r in recs if r["objs"])
    print(f"[{name}] {len(imgs):,} images | {len(recs):,} with labels ({n_pos:,} with fire/smoke, "
          f"{len(recs) - n_pos:,} negatives) | {missing:,} skipped (no label file) | {bad} unreadable labels")
    if imgs and missing > 0.5 * len(imgs):
        print(f"[{name}] WARNING: more than half the images had no label file - check the folder layout / id_map.")
    return _cap(recs, cap, max_neg_frac, seed, name)


def _cap(recs, cap, max_neg_frac, seed, name):
    if not cap or len(recs) <= cap:
        return recs
    rng = random.Random(seed)
    pos = [r for r in recs if r["objs"]]
    neg = [r for r in recs if not r["objs"]]
    rng.shuffle(pos)
    rng.shuffle(neg)
    n_neg = min(len(neg), int(cap * (max_neg_frac if max_neg_frac is not None else len(neg) / len(recs))))
    n_pos = min(len(pos), cap - n_neg)
    n_neg = min(len(neg), cap - n_pos)
    out = pos[:n_pos] + neg[:n_neg]
    print(f"[{name}] capped to {len(out):,} images ({n_pos:,} positive, {n_neg:,} negative)")
    return out


def seq_group(path, block=30):
    """Consecutive video frames (e.g. AoF00123, AoF00124 ...) end up in the same group."""
    stem = Path(path).stem
    m = re.match(r"^(.*?)(\d+)$", stem)
    if not m:
        return stem
    return f"{m.group(1)}#{int(m.group(2)) // block}"


def extract_video_frames(video, out_dir, every_s=2.0, max_frames=300, max_side=960):
    import cv2
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if 1 <= fps <= 240 else 25.0
    step = max(1, int(round(fps * every_s)))
    i, n = 0, 0
    while n < max_frames:
        ok = cap.grab()
        if not ok:
            break
        if i % step == 0:
            ok, frame = cap.retrieve()
            if ok and frame is not None:
                frame = _resize(frame, max_side)
                cv2.imwrite(str(out_dir / f"{Path(video).stem}_f{i:07d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
                n += 1
        i += 1
    cap.release()
    return n


# --------------------------------------------------------------------------------------- staging
def _resize(img, max_side):
    import cv2
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    if s < 1:
        img = cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
    return img


def _stage_worker(args):
    """Read -> resize -> write jpg -> perceptual hash. Runs in a worker process."""
    idx, src, dst, max_side = args
    try:
        import cv2
        import imagehash
        from PIL import Image
        data = np.fromfile(src, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None or img.shape[0] < 16 or img.shape[1] < 16:
            return idx, None, 0, 0
        h0, w0 = img.shape[:2]
        img = _resize(img, max_side)
        cv2.imwrite(dst, img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        small = _resize(img, 256)
        hsh = int(str(imagehash.phash(Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)))), 16)
        return idx, hsh, w0, h0
    except Exception:  # noqa: BLE001
        return idx, None, 0, 0


def _popcount(x):
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(x)
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return (x * np.uint64(0x0101010101010101)) >> np.uint64(56)


class _UF:
    """Union-find with a size cap so long chains of 'similar' images can't merge into one giant group."""

    def __init__(self, n):
        self.p = list(range(n))
        self.size = [1] * n

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b, max_size=None):
        a, b = self.find(a), self.find(b)
        if a == b or (max_size and self.size[a] + self.size[b] > max_size):
            return False
        a, b = min(a, b), max(a, b)
        self.p[b] = a
        self.size[a] += self.size[b]
        return True


def near_pairs(hashes, max_dist, bucket_cap=2000):
    """All pairs (i, j, dist) with Hamming distance <= max_dist (pigeonhole bucketing)."""
    h = np.asarray(hashes, dtype=np.uint64)
    n = len(h)
    k = max_dist + 1
    sizes = [64 // k + (1 if i < 64 % k else 0) for i in range(k)]
    pairs = {}
    off = 0
    for sz in sizes:
        key = (h >> np.uint64(off)) & np.uint64((1 << sz) - 1)
        off += sz
        order = np.argsort(key, kind="stable")
        ks = key[order]
        bounds = np.flatnonzero(np.diff(ks)) + 1
        for grp in np.split(order, bounds):
            m = len(grp)
            if m < 2 or m > bucket_cap:
                continue
            hb = h[grp]
            for s in range(0, m, 1024):
                blk = hb[s:s + 1024]
                d = _popcount(blk[:, None] ^ hb[None, :])
                ii, jj = np.nonzero(d <= max_dist)
                for a, b in zip(ii + s, jj):
                    if a < b:
                        i, j = int(grp[a]), int(grp[b])
                        key2 = (i, j) if i < j else (j, i)
                        pairs[key2] = int(d[a - s, b])
    return pairs


# --------------------------------------------------------------------------------------- build
def build_dataset(records, out_dir, stage_dir, max_side=960, val_frac=0.08, test_frac=0.04,
                  dup_dist=3, group_dist=8, seed=0, workers=None, max_group=300):
    """Stage + hash + dedup + group split + write YOLO dataset. Returns stats dict."""
    t0 = time.time()
    out_dir, stage_dir = Path(out_dir), Path(stage_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    n = len(records)
    print(f"Staging {n:,} images with {workers} processes (resize to <= {max_side}px + perceptual hash) ...")
    jobs = [(i, r["path"], str(stage_dir / f"{i:07d}.jpg"), max_side) for i, r in enumerate(records)]
    hashes = [None] * n
    sizes = [(0, 0)] * n
    done = 0
    with Pool(workers) as pool:
        for idx, hsh, w0, h0 in pool.imap_unordered(_stage_worker, jobs, chunksize=32):
            hashes[idx], sizes[idx] = hsh, (w0, h0)
            done += 1
            if done % 5000 == 0 or done == n:
                el = time.time() - t0
                print(f"   staged {done:,}/{n:,} ({100 * done / n:.0f}%)  elapsed {fmt_secs(el)}  "
                      f"ETA {fmt_secs(el / done * (n - done))}")
    ok = [i for i in range(n) if hashes[i] is not None]
    print(f"   {n - len(ok):,} unreadable images dropped")

    # finalize boxes -> normalized (cls_id, cx, cy, w, h)
    labels = {}
    for i in ok:
        r = records[i]
        w0, h0 = sizes[i]
        boxes = []
        for o in r["objs"]:
            c = CLASS_NAMES.index(o[0])
            if r["fmt"] == "yolo":
                cx, cy, bw, bh = o[1:]
            else:
                W, H = r["wh"] or (w0, h0)
                x1, y1, x2, y2 = o[1:]
                cx, cy, bw, bh = (x1 + x2) / 2 / W, (y1 + y2) / 2 / H, (x2 - x1) / W, (y2 - y1) / H
            x1, y1 = max(0.0, cx - bw / 2), max(0.0, cy - bh / 2)
            x2, y2 = min(1.0, cx + bw / 2), min(1.0, cy + bh / 2)
            if x2 - x1 < 0.002 or y2 - y1 < 0.002:
                continue
            boxes.append((c, (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1))
        labels[i] = boxes

    # exact duplicates
    by_hash = defaultdict(list)
    for i in ok:
        by_hash[hashes[i]].append(i)
    uniq = list(by_hash.keys())
    keep_of_hash = {}
    exact_dups = 0
    for hsh, idxs in by_hash.items():
        best = max(idxs, key=lambda i: (len(labels[i]), -i))   # prefer the copy that has labels
        keep_of_hash[hsh] = best
        exact_dups += len(idxs) - 1
    print(f"   exact duplicates removed: {exact_dups:,}")

    # near duplicates (drop) + similar images (same group)
    print(f"   searching near-duplicates among {len(uniq):,} unique hashes ...")
    pairs = near_pairs(uniq, group_dist)
    reps = [keep_of_hash[h] for h in uniq]
    m = len(reps)
    max_group = max(2, min(max_group, m // 100 + 1))   # no group larger than ~1% of the data
    uf_dup, uf_grp = _UF(m), _UF(m)
    for (a, b), d in sorted(pairs.items(), key=lambda kv: kv[1]):   # closest pairs first
        uf_grp.union(a, b, max_size=max_group)
        if d <= dup_dist:
            uf_dup.union(a, b, max_size=50)
    dup_clusters = defaultdict(list)
    for a in range(m):
        dup_clusters[uf_dup.find(a)].append(a)
    final = []
    for members in dup_clusters.values():
        best = max(members, key=lambda a: (len(labels[reps[a]]), -reps[a]))
        final.append(best)
    print(f"   near-duplicates removed: {m - len(final):,}  -> {len(final):,} images kept")

    # groups: near-duplicate graph + filename sequences + explicit groups (e.g. video name)
    uf = _UF(len(final))
    first_of = {}
    for k, a in enumerate(final):
        r = records[reps[a]]
        for key in (("g", uf_grp.find(a)), ("s", r["src"], r["group"] or seq_group(r["path"]))):
            if key in first_of:
                uf.union(first_of[key], k, max_size=max_group)
            else:
                first_of[key] = k
    groups = defaultdict(list)
    for k in range(len(final)):
        groups[uf.find(k)].append(k)

    rng = random.Random(seed)
    glist = list(groups.values())
    rng.shuffle(glist)
    total = len(final)
    split_of = {}
    n_val = n_test = 0
    for g in glist:
        forced = records[reps[final[g[0]]]]["force_split"]
        if forced:
            sp = forced
        elif n_test < test_frac * total:
            sp = "test"
            n_test += len(g)
        elif n_val < val_frac * total:
            sp = "val"
            n_val += len(g)
        else:
            sp = "train"
        for k in g:
            split_of[k] = sp

    # write
    for sp in ("train", "val", "test"):
        (out_dir / "images" / sp).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / sp).mkdir(parents=True, exist_ok=True)
    stats = {"per_split": defaultdict(Counter), "per_source": defaultdict(Counter)}
    with open(out_dir / "manifest.csv", "w", newline="") as fcsv:
        wr = csv.writer(fcsv)
        wr.writerow(["name", "split", "source", "orig_path", "n_fire", "n_smoke"])
        for k, a in enumerate(final):
            i = reps[a]
            r, sp = records[i], split_of[k]
            name = f"{r['src']}_{i:07d}"
            shutil.move(str(stage_dir / f"{i:07d}.jpg"), str(out_dir / "images" / sp / f"{name}.jpg"))
            lines = [f"{c} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}" for c, cx, cy, bw, bh in labels[i]]
            (out_dir / "labels" / sp / f"{name}.txt").write_text("\n".join(lines))
            nf = sum(1 for b in labels[i] if b[0] == 0)
            ns = sum(1 for b in labels[i] if b[0] == 1)
            wr.writerow([name, sp, r["src"], r["path"], nf, ns])
            for c in (stats["per_split"][sp], stats["per_source"][r["src"]]):
                c["images"] += 1
                c["fire_boxes"] += nf
                c["smoke_boxes"] += ns
                c["images_with_fire"] += int(nf > 0)
                c["images_with_smoke"] += int(ns > 0)
                c["negative_images"] += int(not labels[i])
    shutil.rmtree(stage_dir, ignore_errors=True)
    write_data_yaml(out_dir)
    stats = {k: {kk: dict(vv) for kk, vv in v.items()} for k, v in stats.items()}
    stats.update(exact_duplicates=exact_dups, near_duplicates=m - len(final), groups=len(glist),
                 build_time_s=round(time.time() - t0, 1))
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    print(f"Dataset written to {out_dir} in {fmt_secs(time.time() - t0)}")
    return stats


def write_data_yaml(out_dir, extra_train_dirs=()):
    out_dir = Path(out_dir)
    train = ["images/train"] + list(extra_train_dirs)
    data = {"path": str(out_dir.resolve()), "train": train if len(train) > 1 else train[0], "val": "images/val",
            "test": "images/test", "names": dict(enumerate(CLASS_NAMES))}
    save_yaml(data, out_dir / ("data.yaml" if not extra_train_dirs else "data_v2.yaml"))
    return out_dir / ("data.yaml" if not extra_train_dirs else "data_v2.yaml")


def prepare_all(cfg, raw_dir, work_dir, out_dir):
    """Download + ingest every enabled source and build the merged dataset."""
    raw_dir, work_dir = Path(raw_dir), Path(work_dir)
    ex = work_dir / "extract"
    d = cfg["data"]
    seed = d.get("seed", 0)
    records = []
    s = d["sources"]
    if s.get("dfire", {}).get("enabled", True):
        root = get_dfire(raw_dir, ex)
        if root:
            records += ingest_folder("dfire", root, id_map=_id_map_for(root, DFIRE_ID_MAP, "dfire"), cap=s["dfire"].get("cap"),
                                     max_neg_frac=s["dfire"].get("max_neg_frac"), seed=seed)
    if s.get("fasdd", {}).get("enabled", True):
        root = get_fasdd(raw_dir, ex)
        if root:
            records += ingest_folder("fasdd", root, id_map=_id_map_for(root, FASDD_ID_MAP, "fasdd"), cap=s["fasdd"].get("cap"),
                                     max_neg_frac=s["fasdd"].get("max_neg_frac"), seed=seed)
    for e in d.get("extra_sources", []) or []:
        root = _manual_drop(raw_dir, e["name"], ex) or (Path(e["path"]) if e.get("path") else None)
        if root and Path(root).exists():
            idm = {int(k): v for k, v in e["id_map"].items()} if e.get("id_map") else None
            records += ingest_folder(e["name"], root, id_map=idm, cap=e.get("cap"),
                                     max_neg_frac=e.get("max_neg_frac"), seed=seed)
        else:
            print(f"[{e['name']}] not found - skipped (put it in raw/{e['name']}/)")
    sc = d.get("street") or {}
    if sc.get("enabled"):
        fr = get_street_frames(raw_dir, sc.get("n_train", 2000), sc.get("n_mine", 800), sc.get("n_test", 400), seed=seed)
        if "train" in fr:
            records += ingest_folder("street", fr["train"], negatives_only=True, force_split="train",
                                     group_fn=lambda p: p.name.split("__")[0])
        else:
            print("street frames: not available - training without them")
    neg_dir = raw_dir / d.get("negative_images_dir", "negatives")
    if neg_dir.exists():
        records += ingest_folder("negimg", neg_dir, negatives_only=True, force_split="train",
                                 group_fn=lambda p: p.parent.name)
    vid_dir = raw_dir / d.get("normal_videos_dir", "normal_videos")
    vids = list_files(vid_dir, VID_EXTS)
    if vids:
        fdir = ex / "normal_video_frames"
        print(f"Extracting frames from {len(vids)} normal videos (every {d.get('normal_video_every_s', 2.0)} s) ...")
        for v in vids:
            vd = fdir / v.stem
            if not (vd / ".done").exists():
                extract_video_frames(v, vd, d.get("normal_video_every_s", 2.0), d.get("normal_video_max_frames", 300),
                                     d.get("max_side", 960))
                (vd / ".done").touch()
        records += ingest_folder("negvid", fdir, negatives_only=True, force_split="train",
                                 group_fn=lambda p: p.parent.name)
    if not records:
        raise RuntimeError("No training data found. Check the download messages above.")
    return build_dataset(records, out_dir, work_dir / "stage", max_side=d.get("max_side", 960),
                         val_frac=d.get("val_frac", 0.08), test_frac=d.get("test_frac", 0.04),
                         dup_dist=d.get("dup_dist", 3), group_dist=d.get("group_dist", 8), seed=seed)
