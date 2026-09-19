"""Frozen two-cohort study, paired live inference, and exact-request archival replay."""
from pathlib import Path
import hashlib,json,time
import numpy as np
import pandas as pd
from .study import save_json
from .update_experiment import prepare,generate_cases,run_numerical,assumption_stress,metrics
from .update_agents import LlamaDecisionClient,make_prompt,strict_parse
from .update_graph import build_defense_graph,run_graph
from .core import CHANNELS as ATTACK_CHANNELS

COHORTS=['phase_1','phase_2']
MODES=[('llama_one_pass',1,False),('llama_langgraph',2,True)]

def prepare_cohort(code,run,cohort,recollect=False):
    code=Path(code);cfg=json.loads((code/'protocols'/f'{cohort}.json').read_text())
    return prepare(code,Path(run)/cohort,collect=recollect,protocol_path=code/'protocols'/f'{cohort}.json',data_path=code/'teaching_data'/cohort)

def selected_cases(ctx,cohort):
    """Predefined balanced subset. Selection never reads method outputs or truth errors."""
    ci=COHORTS.index(cohort);severities=[x['name'] for x in ctx['config']['attack_severities']]
    for c in generate_cases(ctx):
        if c['window'] not in [1,9]:continue
        if c['attack'] is None:yield c;continue
        wi=[1,9].index(c['window']);si=severities.index(c['severity'])
        selected_building=(1+wi+ci+si)%ctx['tools'].B
        if c['building']==selected_building:yield c

class ReplayClient:
    """Replay authentic received model text ONLY when the entire request prompt matches.

    This regenerates decisions and repairs; it is not fresh inference or a latency run.
    Missing or changed evidence is an error, never a fabricated model completion.
    """
    source='llama_archived_replay'
    def __init__(self,archive):self.archive=Path(archive)
    def decide(self,stage,evidence,path):
        p=Path(path);original=self.archive/p.parent.name/p.name
        r=json.loads(original.read_text())
        prompt=make_prompt(stage,evidence)
        if r['request']['messages'][0]['content']!=prompt:raise ValueError(f'Replay evidence/prompt mismatch: {original}')
        if r.get('raw') is not None:
            try:parsed=strict_parse(r['raw'],stage);error=None
            except (TypeError,ValueError) as exc:parsed=None;error=str(exc)
            assert parsed==r.get('parsed')
        else:parsed=None;error=r.get('error')
        rec={**r,'source':self.source,'original_source':r['source'],'parsed':parsed,'error':error,'replayed_latency_seconds':r['latency_seconds']}
        save_json(p,rec);return rec

def evaluate_llama(ctx,cohort,archive=None,model='llama3.2:3b',url='http://127.0.0.1:11434'):
    root=ctx['run']/'live_llama';root.mkdir(exist_ok=True);rows=[]
    if (root/'completion.json').exists():raise FileExistsError('Use a fresh run folder.')
    if archive is None:
        live=LlamaDecisionClient(url,model,timeout=600)
        manifest=live.preflight();manifest['runtime']=live.request('/api/version')
        save_json(root/'model_manifest.json',manifest)
    cases=list(selected_cases(ctx,cohort))
    save_json(root/'selected_cases.json',[{k:c[k] for k in ['case_id','attack','building','window','severity']} for c in cases])
    for i,c in enumerate(cases):
        for name,attempts,inspection in MODES:
            client=live if archive is None else ReplayClient(Path(archive)/cohort/'live_llama'/name/'logs')
            start=time.perf_counter();g=build_defense_graph(ctx['tools'],client,root/name/'logs',attempts,inspection)
            result=run_graph(g,c['received'],c['case_id'],root/name/'states');wall=time.perf_counter()-start
            d=result.get('diagnosis',{});b=c['building'];attack=c['attack']
            logs=[json.loads(p.read_text()) for p in (root/name/'logs'/c['case_id']).glob('*.json')]
            r={k:c[k] for k in ['case_id','attack','window','severity']}
            r.update(cohort=cohort,method=name,terminal=result['terminal'],attempts=result['attempts'],inspections=result['inspections'],selected_building=d.get('building',0),selected_channel=d.get('channel','none'),true_building=None if b is None else b+1,record_alarm=result['evidence']['alarm'],joint_location_channel=bool(b is not None and d.get('building')==b+1 and d.get('channel')==ATTACK_CHANNELS[attack]),calls=len(logs),invalid_calls=sum(v.get('error') is not None for v in logs),input_tokens=sum(v.get('input_tokens') or 0 for v in logs),output_tokens=sum(v.get('output_tokens') or 0 for v in logs),model_seconds=sum(v['latency_seconds'] for v in logs),graph_wall_seconds=wall,decision_source=client.source)
            r.update(metrics(result['output'],c['received'],c['truth'],ctx,attack,b));rows.append(r)
            pd.DataFrame(rows).to_csv(root/'metrics.csv',index=False)
        print(f'{cohort} genuine paired subset {i+1}/{len(cases)}: {c["case_id"]}',flush=True)
    save_json(root/'completion.json',{'all_predefined_subset_cases_complete':True,'attack_records':sum(c['attack'] is not None for c in cases),'clean_records':sum(c['attack'] is None for c in cases),'paired_modes':2,'source':'live_inference' if archive is None else 'authentic_request_matched_replay','full_numerical_benchmark':False})
    return pd.DataFrame(rows)

def summarize_study(run):
    run=Path(run)
    all_num=pd.concat([pd.read_csv(run/c/'metrics.csv').assign(cohort=c) for c in COHORTS],ignore_index=True)
    all_num.to_csv(run/'all_numerical_metrics.csv',index=False)
    a=all_num[all_num.attack.notna()]
    agg=dict(cases=('case_id','size'),improved=('improved','sum'),harmed=('harmed','sum'),unchanged=('unchanged','sum'),joint_location_channel=('joint_location_channel','sum'),mean_error_before=('error_before','mean'),mean_error_after=('error_after','mean'),mean_collateral_error=('collateral_error','mean'))
    a.groupby('method').agg(**agg).to_csv(run/'numerical_summary.csv')
    a.groupby(['method','attack']).agg(**agg).to_csv(run/'channel_summary.csv')
    all_num[all_num.attack.isna()].groupby('method').agg(records=('case_id','size'),alarms=('record_alarm','sum'),harmed=('harmed','sum')).to_csv(run/'clean_summary.csv')
    if all((run/c/'live_llama/completion.json').exists() for c in COHORTS):
        lm=pd.concat([pd.read_csv(run/c/'live_llama/metrics.csv') for c in COHORTS],ignore_index=True)
        lm.to_csv(run/'all_llama_metrics.csv',index=False)
        keys=lm[['cohort','case_id']].drop_duplicates()
        rules=all_num[all_num.method=='guarded_rules'].merge(keys,on=['cohort','case_id'])
        paired=pd.concat([rules,lm],ignore_index=True);paired.to_csv(run/'paired_metrics.csv',index=False)
        paired[paired.attack.notna()].groupby('method').agg(**agg).to_csv(run/'paired_summary.csv')
        lm.groupby('method').agg(calls=('calls','sum'),invalid_calls=('invalid_calls','sum'),inspections=('inspections','sum'),replans=('attempts',lambda s:int((s>1).sum())),input_tokens=('input_tokens','sum'),output_tokens=('output_tokens','sum'),total_model_seconds=('model_seconds','sum'),median_record_seconds=('model_seconds','median')).to_csv(run/'llama_costs.csv')
    return all_num
