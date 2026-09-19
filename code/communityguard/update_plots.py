"""Paper figures use freshly evaluated arrays, never historical target curves."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from .case_reporting import CHANNELS,LABELS

COLORS={'truth':'#243b53','attack':'#d1495b','repair':'#007f73'}

def workflow(folder):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots(figsize=(10.2,3.5));ax.set_xlim(0,10.2);ax.set_ylim(0,3.5);ax.axis('off')
    def box(x,y,w,h,text,color):
        ax.add_patch(FancyBboxPatch((x-w/2,y-h/2),w,h,boxstyle='round,pad=0.035',facecolor=color,edgecolor='#52616b',linewidth=.9))
        ax.text(x,y,text,ha='center',va='center',fontsize=11.5)
    box(1.15,2.85,2,.62,'1  Detection\nNumerical evidence','#e8f1f8')
    box(3.8,2.85,2,.62,'2  Incident diagnosis\nRules / Llama','#eee9f7')
    box(6.5,2.85,2,.62,'3  Repair planning\nRules / Llama','#eee9f7')
    box(9.1,2.85,1.9,.62,'4  Verification\nGuard each edit','#e4f2ef')
    box(3.8,1.45,2,.65,'Peer inspection\nOne extra request','#f4f6f8')
    box(6.5,.55,2,.62,'Abstain / roll back\nKeep received data','#fff1e6')
    box(9.1,.55,1.9,.62,'Commit repair\nSave data + trace','#e4f2ef')
    def arrow(a,b,label=None,offset=(0,0)):
        ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','lw':1.1,'color':'#52616b'})
        if label:ax.text((a[0]+b[0])/2+offset[0],(a[1]+b[1])/2+offset[1],label,fontsize=10,ha='center',va='center',backgroundcolor='white')
    arrow((2.18,2.85),(2.77,2.85));arrow((4.83,2.85),(5.47,2.85));arrow((7.53,2.85),(8.12,2.85))
    arrow((3.55,2.49),(3.55,1.82),'request',(-.32,0));arrow((4.1,1.82),(4.1,2.49),'evidence',(.36,0))
    arrow((9.1,2.49),(9.1,.91),'pass',(.30,0))
    arrow((6.5,2.49),(6.5,.91),'abstain',(-.48,0))
    ax.plot([8.6,8.6,6.9],[2.49,1.7,1.7],color='#9c4b30',lw=1.1)
    arrow((6.9,1.7),(6.9,2.49))
    ax.text(7.75,1.94,'reject: retry once',fontsize=10,ha='center',color='#9c4b30',backgroundcolor='white')
    ax.plot([8.6,8.6,7.8,7.8],[1.7,1.2,1.2,.55],color='#52616b',lw=1.1)
    arrow((7.8,.55),(7.53,.55))
    ax.text(8.08,.93,'budget\nused',fontsize=10,ha='center',backgroundcolor='white')
    ax.text(1.2,.75,'LangGraph manages state,\nrouting, and event logs.\nNo clean truth in decisions.',fontsize=10.5,ha='center',va='center')
    fig.tight_layout(pad=.2)
    for ext in ['pdf','png']:fig.savefig(folder/f'workflow.{ext}',dpi=200,bbox_inches='tight',metadata={'Creator':'CommunityGuard update'})
    plt.close(fig)

def plot_update(ctx):
    root=ctx['run'];out=root/'figures';out.mkdir(exist_ok=True)
    workflow(out)
    examples=json.loads((root/'fixed_examples.json').read_text());tools=ctx['tools']
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    for attack,channel,label in zip(['market_hack','meter_hack','inverter_hack','time_spoofing'],CHANNELS,LABELS):
        a=np.load(root/f'example_{attack}.npz');k=CHANNELS.index(channel);col=2*tools.M+tools.cols[k];H=72
        fig,(ax,err)=plt.subplots(2,1,figsize=(4.0,3.2),sharex=True,gridspec_kw={'height_ratios':[2.2,1]},constrained_layout=True)
        h=np.arange(H);truth=a['truth'][:H,col];x=a['received'][:H,col];y=a['guarded_rules'][:H,col]
        ax.plot(h,truth,color=COLORS['truth'],lw=1.7,label='Clean reference')
        ax.plot(h,x,color=COLORS['attack'],lw=1.1,ls='--',label='Received')
        ax.plot(h,y,color=COLORS['repair'],lw=1,ls=':',marker='o',ms=2,markevery=6,label='Guarded rules')
        units='Hour (1–24)' if k==3 else 'Price (currency/kWh)' if k==0 else 'Energy (kWh/step)'
        ax.set_ylabel(units);ax.grid(alpha=.16)
        pre=np.abs(x-truth);post=np.abs(y-truth)
        if k==3:pre=np.abs((x-truth+12)%24-12);post=np.abs((y-truth+12)%24-12)
        err.plot(h,pre,color=COLORS['attack'],ls='--',lw=1.1,label='Before');err.plot(h,post,color=COLORS['repair'],lw=1.3,label='After')
        err.set_ylabel('Abs. error');err.set_xlabel('Hours within first test week');err.set_xlim(0,H-1);err.grid(alpha=.16)
        ax.legend(loc='upper center',bbox_to_anchor=(.5,1.22),ncol=3,fontsize=8.5,frameon=False)
        err.legend(loc='upper right',fontsize=8,frameon=False,ncol=2)
        r=examples[attack];selection=f'selected B{r["selected_building"]}' if r['selected_building'] else 'abstained'
        ax.set_title(f'{label}: injected B3, {selection}',fontsize=9,pad=3)
        for ext in ['pdf','png']:fig.savefig(out/f'{attack}.{ext}',dpi=200,bbox_inches='tight')
        plt.close(fig)
    d=pd.read_csv(root/'metrics.csv');g=d[(d.method=='guarded_rules')&d.attack.notna()]
    counts=g.groupby('attack')[['improved','unchanged','harmed']].sum().reindex(['market_hack','meter_hack','inverter_hack','time_spoofing'])
    fig,ax=plt.subplots(figsize=(7,3),constrained_layout=True);left=np.zeros(4)
    for c,color in [('improved','#007f73'),('unchanged','#9ba8b5'),('harmed','#d1495b')]:
        v=counts[c].to_numpy();ax.barh(LABELS,v,left=left,label=c.capitalize(),color=color)
        for i,(l,n) in enumerate(zip(left,v)):
            if n:ax.text(l+n/2,i,str(n),ha='center',va='center',color='white',fontsize=9)
        left+=v
    ax.set_xlabel('Number of evaluated attack records');ax.legend(ncol=3,loc='upper center',bbox_to_anchor=(.5,1.18),frameon=False);ax.set_xlim(0,counts.sum(1).max()*1.02)
    fig.savefig(out/'tutorial_all_cases.png',dpi=170);plt.close(fig)
    return out
