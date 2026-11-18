from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator, PercentFormatter

D = Path(__file__).resolve().parent
R = D.parent
NAVY='#152D44'
INK='#28445A'
MUTED='#49657A'
GRID='#DFE8EF'
TEAL='#007F86'
BLUE='#4473A7'
ORANGE='#C55337'
PURPLE='#73559B'
COLORS=[TEAL,ORANGE,BLUE,PURPLE]
PAIRS=['Run 2 + original verifier','Run 3 + final verifier']
CM=LinearSegmentedColormap.from_list('fire_smoke',['#F1F7FA','#A6D9E2','#208696','#073954'])
CM.set_bad('#E6EBEF')
plt.rcParams.update({
    'font.family':'DejaVu Sans','font.size':10,'text.color':INK,'axes.labelcolor':INK,
    'axes.titlecolor':NAVY,'axes.titlesize':12,'axes.titleweight':'bold','axes.spines.top':False,
    'axes.spines.right':False,'axes.spines.left':False,'axes.spines.bottom':False,
    'xtick.color':MUTED,'ytick.color':MUTED,'grid.color':GRID,
    'grid.linewidth':0.75,'axes.axisbelow':True,'figure.facecolor':'white',
    'savefig.facecolor':'white','legend.frameon':False,'legend.fontsize':9
})

def csv(p):
    return pd.read_csv(R/p)

def layout(title, subtitle, rows=1, cols=1, wh=(10,5.4), legend=False, b=.20, wspace=.23, hspace=.53):
    f,axs=plt.subplots(rows,cols,figsize=wh,squeeze=False)
    f.subplots_adjust(left=.10,right=.956,top=(.66 if legend else .73) if rows==1 else (.73 if legend else .765),
                      bottom=b,wspace=wspace,hspace=hspace)
    f.text(.04,.965,'FIRE AND SMOKE DETECTION  /  EXPERIMENT RESULTS',va='top',fontsize=9,color=TEAL,weight='bold')
    f.text(.04,.905,title,va='top',fontsize=22,weight='bold',color=NAVY)
    f.text(.041,.838,subtitle,va='top',fontsize=10.2,color=MUTED)
    return f,axs.ravel()

def finish(f,name,source,detail=''):
    f.add_artist(plt.Line2D([.04,.96],[.108,.108],transform=f.transFigure,color=GRID,lw=1))
    if detail: f.text(.04,.082,detail,fontsize=9.1,color=INK,va='top')
    f.text(.04,.037,'Source  '+source,fontsize=8.1,color=MUTED,va='top')
    f.savefig(D/(name+'.png'),dpi=210)
    plt.close(f)

def top_legend(f, labels, colors=COLORS, y=.786):
    f.legend([Patch(facecolor=c) for c in colors[:len(labels)]],labels,loc='upper left',
             bbox_to_anchor=(.039,y),ncols=len(labels),borderaxespad=0,handlelength=1.25,columnspacing=1.8)

def adjust(ax,pct=False):
    ax.grid(axis='y')
    ax.tick_params(axis='both',length=0,pad=6)
    if pct:
        ax.yaxis.set_major_formatter(PercentFormatter(100,decimals=0))

def bars(ax, cats, sets, maxval, *, colors=COLORS,pct=False,dec=0):
    x=np.arange(len(cats))
    n=len(sets)
    width=min(.66/n,.56)
    for i,(s,c) in enumerate(zip(sets,colors)):
        v=np.asarray(s,dtype=float)
        assert len(v)==len(cats) and np.isfinite(v).all()
        bb=ax.bar(x+(i-(n-1)/2)*width, v, width*.87, color=c,zorder=3)
        labels=[f'{a:,.{dec}f}' for a in v]
        ax.bar_label(bb,labels=labels,padding=3,fontsize=8.0,color=NAVY)
    ax.set_xticks(x,cats)
    ax.set_ylim(0,maxval)
    adjust(ax,pct)
    if not pct: ax.yaxis.set_major_locator(MaxNLocator(integer=True))

def lines(ax,epochs,curves,*, labels, colors=COLORS,pct=False,style=None,mark=None):
    for i,(v,label,c) in enumerate(zip(curves,labels,colors)):
        ax.plot(epochs,v,label=label,color=c,lw=1.8,marker='o',ms=3.3,ls=(style or ['-']*len(labels))[i])
    if mark is not None:
        ax.axvline(mark,lw=1,ls=':',color=MUTED,alpha=.8)
    ax.set_xlabel('Completed epoch')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True,nbins=7))
    ax.set_xlim(float(min(epochs))-.22,float(max(epochs))+.22)
    adjust(ax,pct)

def title(ax,t): ax.set_title(t,loc='left',pad=9)

def matrix(ax,m, rows,cols,norm):
    img=ax.imshow(np.ma.masked_invalid(m),cmap=CM,norm=norm)
    for (i,j),v in np.ndenumerate(m):
        ax.text(j,i,'–' if np.isnan(v) else f'{v:,.0f}',ha='center',va='center',weight='bold',
                color='white' if np.isfinite(v) and norm(v)>.56 else NAVY,fontsize=10.4)
    ax.set_xticks(range(len(cols)),cols)
    ax.set_yticks(range(len(rows)),rows)
    ax.tick_params(length=0, pad=5)
    return img

d=csv('evaluation/final_model_evaluation.csv').set_index('class').loc[['all','fire','smoke']]
cats=['Precision','Recall','F1','AP50','AP50–95','AP95']
fields=['precision_%','recall_%','F1_%','AP50_%','AP50-95_%','AP95_%']
f,(a,)=layout('Final model metrics by class','Validation split · 3,764 images · Detector 0.20 / verifier 0.70',wh=(11,5.5),legend=True)
top_legend(f,['Overall','Fire','Smoke'],[NAVY,ORANGE,TEAL])
bars(a,cats,[d.loc[k,fields] for k in d.index],105,colors=[NAVY,ORANGE,TEAL],pct=True,dec=1)
finish(f,'01_final_model_metrics','evaluation/final_model_evaluation.csv',
       'The overall bars in the AP columns are mAP across the two classes. AP95 is at a single IoU of 0.95.')

d=csv('evaluation/old_vs_final_pair_validation.csv')
assert len(d)==2 and (d['images']==3764).all()
f,(a,b)=layout('Model-pair performance on the same validation split','Two detector–verifier pairs · 3,764 images · identical confidence cutoffs',wh=(11.8,5.5),cols=2,legend=True,wspace=.34)
top_legend(f,PAIRS)
title(a,'Box metrics')
bars(a,['Precision','Recall','F1','mAP50','mAP50–95'],
     [r[['precision_%','recall_%','F1_%','mAP50_%','mAP50-95_%']] for _,r in d.iterrows()],94,pct=True,dec=1)
title(b,'Clean images flagged')
bars(b,['Out of 1,615 clean images'],[[v] for v in d.clean_images_flagged],86)
finish(f,'02_model_pair_comparison','evaluation/old_vs_final_pair_validation.csv',
       'Recall was nearly unchanged (56.85% to 56.87%). The final pair flagged 72 clean images versus 39 for the preceding pair.')

d=csv('evaluation/threshold_comparison.csv')
f,axs=layout('Threshold sweep: recall rose, but so did background flags','Baseline 0.20 / 0.70 versus candidate 0.15 / 0.35 · detector / verifier cutoffs',
             rows=2,cols=2,wh=(11,7.9),legend=True,b=.18,wspace=.32,hspace=.66)
top_legend(f,['Baseline (retained)','Tuned candidate'],y=.799)
for i,(split,label) in enumerate([('val','Standard validation'),('hard_val','Hard validation')]):
    sub=d[d.split==split].set_index('config')
    cs=['baseline','tuned']
    ax,bb=axs[2*i:2*i+2]
    title(ax,label+' · box metrics')
    bars(ax,['Precision','Recall','F1','Macro F1'],[100*sub.loc[k,['precision','recall','F1','macro_F1']] for k in cs],92,pct=True,dec=1)
    title(bb,label+' · clean images flagged')
    bars(bb,['Flagged images'],[[sub.loc[k,'bg_images_flagged']] for k in cs],140)
finish(f,'03_threshold_comparison','evaluation/threshold_comparison.csv',
       'The candidate was selected on standard validation, but reduced F1 on hard validation. Baseline cutoffs were retained.')

raw=json.loads((D/'detection_confusion_counts.json').read_text())
f,axs=layout('Fire and smoke detection confusion matrices','Box matching at IoU ≥ 0.50 · rows are actual, columns are predicted',
             rows=2,cols=2,wh=(10.8,8.1),b=.205,wspace=.28,hspace=.63)
classes=['Fire','Smoke','BG']
for i,(split,config,caption) in enumerate([
    ('val','baseline','Standard validation · baseline'),
    ('val','tuned','Standard validation · tuned'),
    ('hard_val','baseline','Hard validation · baseline'),
    ('hard_val','tuned','Hard validation · tuned')]):
    m=np.array(raw[f'{split}_{config}'],dtype=float)
    r=d[(d.split==split)&(d.config==config)].iloc[0]
    tp=m[0,0]+m[1,1]
    assert abs(tp/m[:,:2].sum()-r.precision)<.00006
    assert abs(tp/m[:2,:].sum()-r.recall)<.00006
    for j,key in enumerate(['fire','smoke']):
        assert abs(m[j,j]/m[:,j].sum()-r[key+'_P'])<.00006
        assert abs(m[j,j]/m[j,:].sum()-r[key+'_R'])<.00006
    maxv=max(np.nanmax(np.array(raw[f'{split}_{z}'],dtype=float)) for z in ['baseline','tuned'])
    title(axs[i],caption)
    matrix(axs[i],m,classes,classes,PowerNorm(gamma=.65,vmin=0,vmax=maxv))
finish(f,'04_detection_confusion_matrices','Saved threshold-tuning figure and evaluation/threshold_comparison.csv',
       'BG column = missed labeled box. BG row = unmatched prediction. Background-to-background is not defined.')

d=csv('evaluation/verifier_comparison.csv').set_index('model').loc[['existing_verifier','final_verifier']]
f,(a,)=layout('Crop verifier acceptance and rejection','Fixed event-probability cutoff 0.70 · same validation crops for both verifiers',legend=True)
top_legend(f,['Original verifier','Final verifier'])
bars(a,['Fire crops kept','Smoke crops kept','Background crops rejected'],
     [r[['fire_kept_pct','smoke_kept_pct','negative_rejected_pct']] for _,r in d.iterrows()],108,pct=True,dec=2)
finish(f,'05_verifier_comparison','evaluation/verifier_comparison.csv',
       'Sample sizes: 1,818 fire crops, 1,375 smoke crops, and 194 background crops.')

def recover(pct,n):
    choices=[k for k in range(n+1) if abs(100*k/n-pct)<.00500001]
    assert len(choices)==1,(pct,n,choices)
    return choices[0]
f,axs=layout('Crop verifier confusion counts','The verifier is an event gate. It does not assign the final fire or smoke label.',cols=2,wh=(10.3,5.5),b=.21)
for (key,r),ax,cap in zip(d.iterrows(),axs,['Original verifier','Final verifier']):
    nf,ns,nb=[int(r[k]) for k in ['validation_fire_crops','validation_smoke_crops','validation_negative_crops']]
    cf,cs,cb=[recover(r[k],n) for k,n in zip(['fire_kept_pct','smoke_kept_pct','negative_rejected_pct'],[nf,ns,nb])]
    a=np.array([[cf,nf-cf],[cs,ns-cs],[nb-cb,cb]])
    assert list(a.sum(axis=1))==[nf,ns,nb]
    title(ax,cap)
    matrix(ax,a,['Fire crop','Smoke crop','Background crop'],['Accepted','Rejected'],PowerNorm(gamma=.58,vmin=0,vmax=1818))
finish(f,'06_verifier_gate_confusion_counts','evaluation/verifier_comparison.csv',
       'Each count is the unique integer consistent with the saved percentage and the number of crops in its row.')

d=csv('evaluation/six_positive_clips_per_video.csv')
clips=['f1.mp4','f2.mp4','f3.mp4','s1.mp4','s2.mp4','s3.mp4']
pair_names=d.pair.drop_duplicates().tolist()
assert len(pair_names)==2
det=[d[d.pair==k].set_index('video').loc[clips] for k in pair_names]
f,axs=layout('Positive clips: alarm coverage and timing','Six clips with a known expected event label · timestamps measured from clip start',
             rows=2,cols=1,wh=(10.7,8.2),legend=True,b=.18,hspace=.59)
top_legend(f,PAIRS,y=.800)
a,b=axs
title(a,'Expected alarm detected')
ct=[[int(q[q.expected==event].expected_alarm_detected.sum()) for event in ['fire','smoke']] for q in det]
assert ct==[[3,2],[3,1]]
bars(a,['Fire (3 clips)','Smoke (3 clips)'],ct,3.55)
a.set_yticks(range(4))
title(b,'First alarm with the expected label')
for i,(q,c) in enumerate(zip(det,COLORS)):
    x=np.arange(6)+(i-.5)*.24
    for j,(tx,(_,r)) in enumerate(zip(x,q.iterrows())):
        t=r.first_expected_alarm_s
        if pd.notna(t):
            b.scatter(tx,t,c=c,marker='o' if i==0 else 's',s=46,zorder=4)
            b.annotate(f'{t:.2f}',(tx,t),xytext=(0,8),textcoords='offset points',ha='center',fontsize=8.1,color=NAVY)
        else:
            assert not r.expected_alarm_detected
            b.scatter(tx,-.30,c=c,marker='x',s=48,zorder=4)
b.axhline(0,color=GRID,lw=.8)
b.set_xticks(range(6),[k[:-4] for k in clips])
b.set_ylim(-.55,4.65)
b.set_yticks([0,1,2,3,4])
b.set_ylabel('Seconds from clip start')
adjust(b)
finish(f,'07_positive_video_alarms','evaluation/six_positive_clips_per_video.csv',
       '× = expected alarm not raised. On s2, both pairs raised fire instead of smoke.')

d=csv('evaluation/six_traffic_clips_per_video.csv')
clips=[f'r{i}.mp4' for i in range(1,7)]
det=[d[d.pair==k].set_index('video').loc[clips] for k in pair_names]
assert all(q.frames_checked.sum()==711 for q in det)
f,axs=layout('Traffic clips: occasional flags, no false alarms','Six fire-free and smoke-free traffic clips · 711 sampled frames per model pair',cols=2,wh=(12.3,5.5),legend=True,wspace=.27)
top_legend(f,PAIRS)
title(axs[0],'Frames flagged by YOLO')
title(axs[1],'Frames flagged after verification')
for ax,col in zip(axs,['yolo_flagged_frames','verified_flagged_frames']):
    bars(ax,[k[:-4] for k in clips],[q[col] for q in det],3.7)
    ax.set_yticks(range(4))
assert [int(q.total_false_alarms.sum()) for q in det]==[0,0]
finish(f,'08_traffic_video_false_positives','evaluation/six_traffic_clips_per_video.csv',
       'Both pairs raised zero false alarms. After verification, the preceding pair flagged 2 frames and the final pair flagged 3.')

# Individual epoch charts keep every logged epoch, with validation metrics and loss on separate panels.
for i in (1,2,3):
    name=f'training/ft{i}_epochs.csv'
    d=csv(name)
    assert len(d)==[16,20,11][i-1] and np.array_equal(d.epoch,np.arange(1,len(d)+1))
    ep=d.epoch
    select=[1,None,11][i-1]
    split='standard validation (3,764 images after exact-overlap filtering)' if i==3 else 'original validation (3,829 images)'
    note=['Dotted line marks the selected epoch 1 checkpoint.',
          'Every logged epoch is shown. The selected checkpoint was exported using validation fitness.',
          'Dotted line marks epoch 11, the last completed detector epoch and the submitted detector.'][i-1]
    f,(a,b)=layout(f'Fine-tuning run {i}: training history',f'{len(d)} completed epochs · detector-only metrics on {split}',
                    cols=2,wh=(11.2,6.1),b=.265,wspace=.32)
    metrics=['Precision','Recall','mAP50','mAP50–95']
    title(a,'Validation metrics')
    lines(a,ep,[100*d[k] for k in ['val_precision','val_recall','val_map50','val_map50_95']],labels=metrics,pct=True,mark=select)
    a.set_ylim(35,82)
    a.legend(loc='lower center',bbox_to_anchor=(.5,-.285),ncols=2,fontsize=8)
    title(b,'Training loss')
    lines(b,ep,[d[k] for k in ('train_box_loss','train_cls_loss')],labels=['Box loss','Classification loss'],mark=select)
    b.set_ylabel('Loss')
    b.legend(loc='lower center',bbox_to_anchor=(.5,-.24),ncols=2,fontsize=8)
    finish(f,f'{8+i:02d}_fine_tuning_run_{i}_epochs',name,note)

d=csv('training/verifier_epochs.csv')
assert list(d.epoch)==list(range(1,7))
f,(a,)=layout('Verifier fine-tuning by epoch','Retention and rejection rates on the validation crop set at cutoff 0.70',wh=(10,5.4),legend=True)
top_legend(f,['Fire kept','Smoke kept','Background rejected'])
lines(a,d.epoch,[d[k] for k in ('fire_kept_pct','smoke_kept_pct','negative_rejected_pct')],
      labels=['Fire kept','Smoke kept','Background rejected'],pct=True,mark=5)
a.set_ylim(35,103)
finish(f,'12_verifier_epochs','training/verifier_epochs.csv','Dotted line marks epoch 5, the selected verifier checkpoint.')

for i in (1,2):
    name=f'original_model/detector_v{i}_results.csv'
    d=csv(name)
    d.columns=d.columns.str.strip()
    ep=d.epoch
    assert len(d)==[26,10][i-1]
    f,(a,b)=layout(f'Initial detector stage {i}: training history',
                   f'{len(d)} completed epochs · detector-only validation log from the initial training',
                   cols=2,wh=(11.2,6.1),b=.265,wspace=.31)
    title(a,'Validation metrics')
    lines(a,ep,[100*d['metrics/'+key+'(B)'] for key in ('precision','recall','mAP50','mAP50-95')],
          labels=['Precision','Recall','mAP50','mAP50–95'],pct=True)
    a.legend(loc='lower center',bbox_to_anchor=(.5,-.285),ncols=2,fontsize=8)
    title(b,'Training and validation loss')
    lines(b,ep,[d[k] for k in ['train/box_loss','val/box_loss','train/cls_loss','val/cls_loss']],
          labels=['Train box','Validation box','Train class','Validation class'],colors=[TEAL,TEAL,ORANGE,ORANGE],style=['-','--','-','--'])
    b.set_ylabel('Loss')
    b.legend(loc='lower center',bbox_to_anchor=(.5,-.285),ncols=2,fontsize=8)
    finish(f,f'{12+i:02d}_initial_detector_stage_{i}',name,
           'Training curves describe the initial detector. They are not the final detector–verifier evaluation.')
print('Generated 14 PNG charts.')
