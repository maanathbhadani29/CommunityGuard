"""Generate manuscript values, figures, and provenance from completed study outputs."""
from pathlib import Path
import json,hashlib,shutil,subprocess
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .evidence_reporting import save_json
from .case_reporting import CHANNELS,LABELS,plot_cases
from .update_plots import workflow,COLORS
ATTACKS=['market_hack','meter_hack','inverter_hack','time_spoofing']

def figures(code,run):
    code=Path(code);run=Path(run);out=run/'figures';out.mkdir(exist_ok=True);workflow(out)
    plot_cases(code,run)
    d=pd.read_csv(run/'all_numerical_metrics.csv');g=d[(d.method=='guarded_rules')&d.attack.notna()]
    counts=g.groupby('attack')[['improved','unchanged','harmed']].sum().reindex(ATTACKS)
    fig,ax=plt.subplots(figsize=(8,3.3),constrained_layout=True);left=np.zeros(4)
    for c,color in [('improved','#007f73'),('unchanged','#9ba8b5'),('harmed','#d1495b')]:
        v=counts[c].to_numpy();ax.barh(LABELS,v,left=left,label=c.capitalize(),color=color)
        for i,(l,n) in enumerate(zip(left,v)):
            if n>=20:ax.text(l+n/2,i,str(n),ha='center',va='center',color='white',fontsize=10)
        left+=v
    ax.set_xlabel('Attacked records (CommunityGuard-AI with rules, both cohorts)');ax.legend(ncol=3,loc='upper center',bbox_to_anchor=(.5,1.18),frameon=False);ax.set_xlim(0,counts.sum(1).max()*1.03)
    fig.savefig(out/'all_case_recovery.png',dpi=180);plt.close(fig)
    return out


def build_paper(code, run, compile_pdf=True):
    """Summarize and plot a completed study, then produce the scientific report."""
    from .evidence_reporting import refresh_paper
    from .final_study import summarize_study
    code, run = Path(code), Path(run)
    summarize_study(run)
    figures(code, run)
    return refresh_paper(code, run, compile_pdf=compile_pdf)
