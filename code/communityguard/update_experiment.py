"""Fixed-design evaluation. Ground truth enters metrics only after execution."""
from pathlib import Path
import json,random,time
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from .study import collect_dataset,inject_case,save_json
from .pipeline import build_model
from .core import ATTACKS,CHANNELS as ATTACK_CHANNELS,localize,replace_and_verify
from .update_method import RepairTools,CHANNELS,distance
from .update_agents import RulesClient,LlamaDecisionClient
from .update_graph import build_defense_graph,run_graph


def prepare(code,run,collect=False,protocol_path=None,data_path=None):
    code=Path(code);run=Path(run);run.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((Path(protocol_path) if protocol_path else code/'update_protocol.json').read_text())
    data=Path(data_path) if data_path else code/'teaching_data'
    if collect or not (data/'clean_observations.npz').exists():raw,meta=collect_dataset(cfg,data)
    else:
        raw=np.load(data/'clean_observations.npz')['raw'];meta=json.loads((data/'data_metadata.json').read_text())
    n=cfg['train_hours'];c=n+cfg['calibration_hours'];total=c+cfg['test_hours']
    if len(raw)<total: raise ValueError('Bundled rollout is shorter than the requested protocol; recollect into a separate package copy.')
    raw=raw[:total]
    train,cal,test=raw[:n],raw[n:c],raw[c:]
    save_json(run/'protocol.json',cfg)
    tools=RepairTools(meta['observation_names'],meta['buildings'],cfg).fit(train,cal,run/'01_models')
    # Baseline retrained on exactly the same training period.
    scaler=StandardScaler().fit(train)
    z=scaler.transform(train).astype(np.float32)
    torch.manual_seed(cfg['ae_seed']);np.random.seed(cfg['ae_seed']);random.seed(cfg['ae_seed'])
    model,predict,losses,_=build_model(z,cfg['ae_epochs'],cfg['ae_learning_rate'],cfg['threads'])
    torch.save(model.state_dict(),run/'01_models/legacy_ae.pt')
    np.savez_compressed(run/'01_models/legacy_scaler.npz',mean=scaler.mean_,scale=scaler.scale_,losses=losses)
    metric_scale=np.maximum(np.quantile(np.abs(train),.95,axis=0),.01)
    for b in range(meta['buildings']):
        for name,v in [('hour',24),('month',12),('day_type',8)]:metric_scale[b*tools.M+tools.names.index(name)]=v
    return {'code':code,'run':run,'config':cfg,'raw':raw,'metadata':meta,'train':train,'calibration':cal,'test':test,'tools':tools,'legacy_scaler':scaler,'legacy_predict':predict,'metric_scale':metric_scale}


def metrics(output,received,truth,ctx,attack=None,b=None):
    tools=ctx['tools'];scale=ctx['metric_scale']
    err=output.astype(float)-truth.astype(float);pre=received.astype(float)-truth.astype(float)
    for i in range(tools.B):
        j=i*tools.M+tools.names.index('hour');err[:,j]=distance(output[:,j],truth[:,j],3);pre[:,j]=distance(received[:,j],truth[:,j],3)
    changed=np.abs(received-truth)>1e-7
    squared=(err/scale)**2;before=float(np.mean((pre/scale)**2));after=float(squared.mean())
    r={'error_before':before,'error_after':after,'improved':after<before-1e-12,'harmed':after>before+1e-12,'unchanged':abs(after-before)<=1e-12,'collateral_error':float(np.where(~changed,squared,0).mean()),'changed_entries':int(np.count_nonzero(np.abs(output-received)>1e-7))}
    if attack is not None:
        k=CHANNELS.index(ATTACK_CHANNELS[attack]);col=b*tools.M+tools.cols[k]
        r['target_rmse_before']=float(np.sqrt(np.mean(pre[:,col]**2)))
        r['target_rmse_after']=float(np.sqrt(np.mean(err[:,col]**2)))
    return r


def generate_cases(ctx):
    cfg=ctx['config'];W=cfg['window_hours'];test=ctx['test'];names=ctx['tools'].names
    for w in range(len(test)//W):
        clean=test[w*W:(w+1)*W]
        yield {'case_id':f'clean_w{w+1}','attack':None,'building':None,'window':w+1,'severity':'clean','truth':clean,'received':clean.copy()}
        for si,par in enumerate(cfg['attack_severities']):
            for ai,attack in enumerate(ATTACKS):
                for b in range(ctx['tools'].B):
                    seed=cfg['attack_seed']+10000*w+1000*si+100*ai+b
                    x=inject_case(clean,attack,b,names,seed,par)
                    yield {'case_id':f'{par["name"]}_{attack}_b{b+1}_w{w+1}','attack':attack,'building':b,'window':w+1,'severity':par['name'],'truth':clean,'received':x}


def run_numerical(ctx):
    run=ctx['run'];tools=ctx['tools'];rows=[];examples={};coverage=[]
    for ix,case in enumerate(generate_cases(ctx)):
        x=case['received'];truth=case['truth'];b=case['building'];attack=case['attack']
        outputs={'no_intervention':x}
        z=ctx['legacy_scaler'].transform(x).astype(np.float32)
        loc=localize(z,ctx['legacy_predict'](z),tools.B,tools.names)
        original=replace_and_verify(z,ctx['legacy_predict'],loc['building_index'],tools.M,'DIGITAL_TWIN_OVERRIDE')
        outputs['original_block']=ctx['legacy_scaler'].inverse_transform(original['defended'])
        ab,ak,ae=tools.ae_localization(x);target=x.copy();col=ab*tools.M+tools.cols[ak];target[:,col]=ae[:,ab,ak]
        outputs['typed_ae_only']=target
        # Per-case graph/checkpointer keeps memory bounded. Trace files preserve events.
        graph=build_defense_graph(tools,RulesClient(),run/'02_agent_logs',max_attempts=ctx['config']['max_attempts'])
        result=run_graph(graph,x,case['case_id'],run/'03_graph_states')
        outputs['guarded_rules']=result['output']
        d=result.get('diagnosis',{})
        for method,y in outputs.items():
            r={k:case[k] for k in ['case_id','attack','window','severity']};r['true_building']=None if b is None else b+1
            rb=loc['building_index']+1 if method=='original_block' else ab+1 if method=='typed_ae_only' else d.get('building',0) if method=='guarded_rules' else 0
            rk=CHANNELS[ak] if method=='typed_ae_only' else d.get('channel','none') if method=='guarded_rules' else 'block' if method=='original_block' else 'none'
            r.update(method=method,selected_building=rb,selected_channel=rk,localized_correctly=bool(b is not None and rb==b+1),joint_location_channel=bool(b is not None and rb==b+1 and rk==ATTACK_CHANNELS[attack]),record_alarm=result['evidence']['alarm'] if method=='guarded_rules' else True if method!='no_intervention' else False,terminal=result['terminal'] if method=='guarded_rules' else 'unconditional',operation=result.get('proposal',{}).get('operation','NONE') if method=='guarded_rules' else 'baseline')
            r.update(metrics(y,x,truth,ctx,attack,b));rows.append(r)
        if attack is None:
            q=tools.references(x);z=tools.channels(truth)
            for k in range(4):coverage.append({'window':case['window'],'channel':CHANNELS[k],'empirical_reference_coverage':float(np.mean(np.abs(distance(z[:,:,k],q[:,:,k],k))<=tools.delta[None,:,k]))})
        if case['window']==1 and case['severity']=='original' and b==2:
            np.savez_compressed(run/f'example_{attack}.npz',truth=truth,received=x,**outputs)
            examples[attack]={'case_id':case['case_id'],'selected_building':d.get('building',0),'selected_channel':d.get('channel','none'),'terminal':result['terminal'],'operation':result.get('proposal',{}).get('operation','NONE')}
        if ix%(1+len(ctx['config']['attack_severities'])*4*tools.B)==0:print(f'Evaluated window {case["window"]}/{len(ctx["test"])//ctx["config"]["window_hours"]}',flush=True)
    frame=pd.DataFrame(rows);frame.to_csv(run/'metrics.csv',index=False)
    pd.DataFrame(coverage).to_csv(run/'reference_coverage.csv',index=False)
    attacked=frame[frame.attack.notna()];clean=frame[frame.attack.isna()]
    summary=attacked.groupby('method').agg(cases=('case_id','size'),improved=('improved','sum'),harmed=('harmed','sum'),unchanged=('unchanged','sum'),localized_correctly=('localized_correctly','sum'),joint_location_channel=('joint_location_channel','sum'),mean_error_before=('error_before','mean'),mean_error_after=('error_after','mean'),mean_collateral_error=('collateral_error','mean')).reset_index()
    summary.to_csv(run/'method_summary.csv',index=False)
    attacked.groupby(['method','attack','severity']).agg(cases=('case_id','size'),improved=('improved','sum'),harmed=('harmed','sum'),mean_target_rmse_before=('target_rmse_before','mean'),mean_target_rmse_after=('target_rmse_after','mean')).reset_index().to_csv(run/'attack_summary.csv',index=False)
    clean.groupby('method').agg(records=('case_id','size'),alarms=('record_alarm','sum'),harmed=('harmed','sum'),mean_error_after=('error_after','mean')).reset_index().to_csv(run/'clean_controls.csv',index=False)
    save_json(run/'fixed_examples.json',examples)
    status={'decision_source':'deterministic rules in LangGraph','attack_records':int(attacked[attacked.method=='guarded_rules'].shape[0]),'clean_records':len(clean[clean.method=='guarded_rules']),'test_hours':len(ctx['test']),'cohort':ctx['config']['schema'],'same_operating_windows_reused':True}
    save_json(run/'evidence_status.json',status)
    print(summary.to_string(index=False),flush=True)
    return frame,summary


def run_llama(ctx,model='llama3.1:8b',url='http://127.0.0.1:11434',max_cases=None):
    """Paired one-pass/full-graph ablation on exactly the same ordered records.

    max_cases is a development convenience; truncated runs are NOT the full paper benchmark.
    """
    client=LlamaDecisionClient(url,model);root=ctx['run']/'live_llama';root.mkdir(exist_ok=True)
    if (root/'metrics.csv').exists():raise FileExistsError('Choose a new run directory to preserve previous live results.')
    save_json(root/'model_manifest.json',client.preflight());rows=[]
    for i,case in enumerate(generate_cases(ctx)):
        if max_cases is not None and i>=max_cases:break
        for name,attempts,inspection in [('llama_one_pass',1,False),('llama_langgraph',2,True)]:
            graph=build_defense_graph(ctx['tools'],client,root/name/'logs',attempts,inspection)
            result=run_graph(graph,case['received'],case['case_id'],root/name/'states')
            d=result.get('diagnosis',{});b=case['building'];attack=case['attack']
            r={k:case[k] for k in ['case_id','attack','window','severity']};r.update(method=name,terminal=result['terminal'],attempts=result['attempts'],inspections=result['inspections'],selected_building=d.get('building',0),selected_channel=d.get('channel','none'),joint_location_channel=bool(b is not None and d.get('building')==b+1 and d.get('channel')==ATTACK_CHANNELS[attack]))
            r.update(metrics(result['output'],case['received'],case['truth'],ctx,attack,b));rows.append(r)
            pd.DataFrame(rows).to_csv(root/'metrics.csv',index=False)
    save_json(root/'completion.json',{'full_benchmark':max_cases is None or max_cases>=(len(ctx['test'])//ctx['config']['window_hours'])*(1+2*4*7),'records_evaluated':len(rows)//2,'paired_modes':2,'claims':'Compare recovery, abstention, invalid outputs, latency and tokens. Identical or worse performance must be reported.'})
    return pd.DataFrame(rows)


def assumption_stress(ctx):
    """Out-of-model control: load and protected net meter corrupted together.

    It demonstrates a boundary; these records are separate from the single-channel benchmark.
    """
    rows=[];tools=ctx['tools'];W=ctx['config']['window_hours']
    for w in range(len(ctx['test'])//W):
        truth=ctx['test'][w*W:(w+1)*W];x=truth.copy();b=2
        load=b*tools.M+tools.names.index('non_shiftable_load');net=b*tools.M+tools.names.index('net_electricity_consumption')
        delta=-.3*x[:,load];x[:,load]+=delta;x[:,net]+=delta
        g=build_defense_graph(tools,RulesClient(),ctx['run']/'stress_logs',2)
        r=run_graph(g,x,f'load_plus_net_w{w+1}',ctx['run']/'stress_states')
        rows.append({'window':w+1,'terminal':r['terminal'],**metrics(r['output'],x,truth,ctx,'meter_hack',b)})
    out=pd.DataFrame(rows);out.to_csv(ctx['run']/'assumption_stress.csv',index=False)
    return out
