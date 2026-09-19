"""Fresh-data, held-out evaluation of the CommunityGuard numerical mechanism.

This module evaluates hypotheses; no expected paper curves or target scores enter
training, model selection, localization, replacement, or plotting.
"""
from pathlib import Path
import json, random, platform, importlib.metadata
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from .core import ATTACKS, CHANNELS, localize, replace_and_verify
from .pipeline import build_model, collect


def save_json(path, data):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    def encode(value):
        if isinstance(value,np.ndarray): return value.tolist()
        if isinstance(value,np.generic): return value.item()
        if isinstance(value,Path): return str(value)
        raise TypeError(type(value).__name__)
    path.write_text(json.dumps(data,indent=2,default=encode,allow_nan=False)+'\n',encoding='utf-8')


def collect_dataset(protocol, folder):
    """Advance CityLearn once, with zero actions, and save unmodified observations."""
    from citylearn.citylearn import CityLearnEnv
    folder=Path(folder); folder.mkdir(parents=True,exist_ok=True)
    total=sum(protocol[k] for k in ['train_hours','calibration_hours','test_hours'])
    env=CityLearnEnv(schema=protocol['schema'],central_agent=False)
    names=list(env.observation_names[0]); B=len(env.buildings)
    assert all(list(v)==names for v in env.observation_names)
    raw,_=collect(env,total,names,'baseline',0,np.random.default_rng(0))
    assert raw.shape==(total,B*len(names)) and np.isfinite(raw).all()
    np.savez_compressed(folder/'clean_observations.npz',raw=raw)
    metadata={'schema':protocol['schema'],'citylearn_version':importlib.metadata.version('citylearn'),
              'buildings':B,'features_per_building':len(names),'observation_names':names,
              'hours':total,'source_start_step':0,'actions':'zero at every step',
              'collected_from':'fresh CityLearn simulation; no historical paper arrays loaded'}
    save_json(folder/'data_metadata.json',metadata)
    return raw,metadata


def split_and_scale(raw,protocol):
    n=protocol['train_hours']; c=n+protocol['calibration_hours']
    assert len(raw)==c+protocol['test_hours']
    indices={'train':np.arange(n),'calibration':np.arange(n,c),'test':np.arange(c,len(raw))}
    assert not set(indices['train']) & set(indices['test'])
    assert not set(indices['calibration']) & set(indices['test'])
    scaler=StandardScaler().fit(raw[indices['train']])
    scaled={name:scaler.transform(raw[ix]).astype(np.float32) for name,ix in indices.items()}
    return scaler,indices,scaled


def train_fresh(z,seed,protocol):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    model,predict,losses,norm=build_model(z,protocol['epochs'],protocol['learning_rate'],protocol['threads'])
    return model,predict,losses,norm


def inject_case(raw,attack,building,names,seed,parameters):
    """Modify one raw channel; hour spoofing preserves CityLearn's 1..24 domain."""
    clean=np.asarray(raw,dtype=np.float32)
    x=clean.copy(); col=building*len(names)+names.index(CHANNELS[attack])
    rng=np.random.default_rng(seed);v=clean[:,col]
    if attack=='market_hack': x[:,col]=v*parameters['market_multiplier']+rng.normal(0,parameters['market_noise_sd'],len(v))
    elif attack=='meter_hack': x[:,col]=v*parameters['meter_multiplier']
    elif attack=='inverter_hack': x[:,col]=v*rng.uniform(*parameters['inverter_uniform'],len(v))
    elif attack=='time_spoofing':
        assert np.isin(v,np.arange(1,25)).all()
        x[:,col]=((v-1-parameters['clock_shift_hours'])%24)+1
    else: raise ValueError(attack)
    untouched=np.ones(x.shape[1],dtype=bool);untouched[col]=False
    np.testing.assert_array_equal(x[:,untouched],clean[:,untouched])
    return x


def residuals(x,predict,B,M):
    return ((x-predict(x))**2).reshape(len(x),B,M).mean(axis=2)


def calibrate(clean_calibration,predict,B,M,quantile):
    scores=residuals(clean_calibration,predict,B,M).max(axis=1)
    threshold=float(np.quantile(scores,quantile,method='higher'))
    return threshold,scores


def truth_metrics(y,x,clean,scaler,B,names,attack,building):
    """Evaluation-only truth. Never called by the model or decision rule."""
    error=(y.astype(np.float64)-clean.astype(np.float64))**2
    changed=np.abs(x.astype(np.float64)-clean.astype(np.float64))>1e-7
    raw_y=scaler.inverse_transform(y).reshape(len(y),B,len(names))
    h=raw_y[:,:,names.index('hour')]
    physical=raw_y[:,:,[names.index('non_shiftable_load'),names.index('solar_generation')]]
    out={'clean_mse':float(error.mean()),
         'error_on_attacked_entries':float(np.where(changed,error,0).mean()),
         'collateral_error_contribution':float(np.where(~changed,error,0).mean()),
         'invalid_hour_fraction':float(((h<1-1e-4)|(h>24+1e-4)|(np.abs(h-np.round(h))>1e-4)).mean()),
         'negative_load_pv_fraction':float((physical < -1e-4).mean())}
    assert np.isclose(out['clean_mse'],out['error_on_attacked_entries']+out['collateral_error_contribution'],rtol=1e-10,atol=1e-12)
    if attack is not None:
        j=names.index(CHANNELS[attack]);col=building*len(names)+j
        out['target_channel_mse_standardized']=float(error[:,col].mean())
        raw_clean=scaler.inverse_transform(clean)[:,col]
        delta=raw_y[:,building,j]-raw_clean
        if attack=='time_spoofing': delta=(delta+12)%24-12
        out['target_channel_rmse_native']=float(np.sqrt(np.mean(delta**2)))
    return out


def evaluate_case(case,predict,scaler,B,names,threshold):
    x,clean=case['x'],case['clean'];M=len(names)
    loc=localize(x,predict(x),B,names)
    numerical=replace_and_verify(x,predict,loc['building_index'],M,'DIGITAL_TWIN_OVERRIDE')
    base=numerical['mse_pre'];post=numerical['mse_post']
    pred=numerical['prediction_before'];block=numerical['defended']
    selected_sensor=max(loc['sensors'],key=loc['sensors'].get)
    sensor_channel={'Electricity_Pricing':'electricity_pricing','Non_Shiftable_Load':'non_shiftable_load',
                    'Solar_Generation':'solar_generation','Clock_Sync':'hour'}[selected_sensor]
    channel=x.copy();col=loc['building_index']*M+names.index(sensor_channel)
    channel[:,col]=pred[:,col]
    methods={'no_intervention':x,'block_candidate':block,
             'residual_gated_block':block if numerical['accepted'] else x,'single_channel_candidate':channel}
    if case['attack'] is not None:
        oracle=x.copy();col=case['true_building']*M+names.index(CHANNELS[case['attack']])
        oracle[:,col]=pred[:,col]
        methods['oracle_location_channel']=oracle
    scores=((x-pred)**2).reshape(len(x),B,M).mean(2)
    alarm=scores.max(1)>threshold
    changed_hour=np.any(np.abs(x-clean)>1e-7,axis=1)
    truth_pre=truth_metrics(x,x,clean,scaler,B,names,case['attack'],case['true_building'])['clean_mse']
    common={k:case[k] for k in ['case_id','attack','true_building','window','model_seed']}
    common.update(localized_building=loc['building_index'],peak_hour=loc['peak_hour'],
                  selected_sensor=selected_sensor,threshold=threshold,record_alarm=bool(alarm.any()),
                  changed_hours=int(changed_hour.sum()),hourly_alarm_rate=float(alarm.mean()),
                  localization_correct=(loc['building_index']==case['true_building']) if case['attack'] is not None else None,
                  changed_hour_detection_rate=float(alarm[changed_hour].mean()) if changed_hour.any() else None,
                  changed_hour_joint_detection_localization=float((alarm[changed_hour] & (scores[changed_hour].argmax(1)==case['true_building'])).mean()) if changed_hour.any() else None,
                  original_residual_acceptance=numerical['accepted'],clean_mse_before=truth_pre,residual_mse_before=base)
    rows=[]
    for method,y in methods.items():
        after=float(np.mean((y-predict(y))**2))
        metrics=truth_metrics(y,x,clean,scaler,B,names,case['attack'],case['true_building'])
        rows.append({**common,'method':method,'residual_mse_after':after,**metrics,
                     'clean_recovery_improved':metrics['clean_mse']<truth_pre-1e-10,
                     'clean_recovery_harmed':metrics['clean_mse']>truth_pre+1e-10,
                     'residual_reduced':after<base,'output_changed':bool(np.any(y!=x))})
    return rows,{'methods':methods,'prediction':pred,'location':loc,'building_scores':scores,
                 'alarm':alarm,'changed_hour':changed_hour,'x':x,'clean':clean}


def evaluate_all(raw,metadata,protocol,scaler,indices,models,folder):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    B=metadata['buildings'];names=metadata['observation_names'];W=protocol['window_hours']
    test=raw[indices['test']];assert len(test)%W==0
    rows=[];examples={};clean_diagnostics=[]
    for seed,record in models.items():
        predict=record['predict'];threshold=record['threshold']
        for window in range(len(test)//W):
            raw_clean=test[window*W:(window+1)*W]
            clean=scaler.transform(raw_clean).astype(np.float32)
            clean_case={'case_id':f'clean_w{window+1}_s{seed}','attack':None,'true_building':-1,
                        'window':window+1,'model_seed':seed,'x':clean,'clean':clean}
            clean_rows,clean_details=evaluate_case(clean_case,predict,scaler,B,names,threshold)
            rows.extend(clean_rows)
            clean_diagnostics.append({'seed':seed,'window':window+1,'false_alarm_hours':int(clean_details['alarm'].sum()),
                                      'hours':W,'hourly_false_alarm_rate':float(clean_details['alarm'].mean()),
                                      'record_alarm':bool(clean_details['alarm'].any())})
            for ai,attack in enumerate(ATTACKS):
                for building in range(B):
                    attack_seed=protocol['attack_seed_base']+1000*window+100*ai+building
                    attacked=inject_case(raw_clean,attack,building,names,attack_seed,protocol['attack_parameters'])
                    x=scaler.transform(attacked).astype(np.float32)
                    case={'case_id':f'{attack}_b{building+1}_w{window+1}_s{seed}','attack':attack,'true_building':building,
                          'window':window+1,'model_seed':seed,'x':x,'clean':clean}
                    case_rows,details=evaluate_case(case,predict,scaler,B,names,threshold)
                    for row in case_rows:row['attack_seed']=attack_seed
                    rows.extend(case_rows)
                    case_folder=folder/'arrays';case_folder.mkdir(exist_ok=True)
                    np.savez_compressed(case_folder/(case['case_id']+'.npz'),attacked=x,clean=clean,prediction=details['prediction'],
                                        building_scores=details['building_scores'],**details['methods'])
                    if seed==42 and window==0 and building==2:
                        examples[attack]={'case':case,**details}
    frame=pd.DataFrame(rows)
    frame.to_csv(folder/'all_case_metrics.csv',index=False)
    pd.DataFrame(clean_diagnostics).to_csv(folder/'clean_false_alarms.csv',index=False)
    return frame,examples,pd.DataFrame(clean_diagnostics)


def summarize(frame,clean_false_alarms):
    attacked=frame[frame.attack.notna()]
    original=attacked[attacked.method=='block_candidate']
    by_attack=original.groupby('attack',sort=False).agg(cases=('case_id','count'),
        localization_correct=('localization_correct','sum'),residual_accepted=('original_residual_acceptance','sum'),
        clean_improved=('clean_recovery_improved','sum'),clean_harmed=('clean_recovery_harmed','sum'),
        mean_changed_hour_detection=('changed_hour_detection_rate','mean'),
        mean_joint_detection_localization=('changed_hour_joint_detection_localization','mean'),
        mean_clean_mse_before=('clean_mse_before','mean'),mean_clean_mse_after=('clean_mse','mean'))
    methods=attacked.groupby('method',sort=False).agg(cases=('case_id','count'),mean_clean_mse=('clean_mse','mean'),
        improved=('clean_recovery_improved','sum'),harmed=('clean_recovery_harmed','sum'),
        mean_collateral_error=('collateral_error_contribution','mean'),mean_invalid_hour_fraction=('invalid_hour_fraction','mean'))
    bad=original[original.original_residual_acceptance & original.clean_recovery_harmed]
    summary={'attack_records':len(original),'unique_corrupted_records':int(original[['attack','true_building','window']].drop_duplicates().shape[0]),
             'model_seeds':int(original.model_seed.nunique()),'correct_localizations':int(original.localization_correct.sum()),
             'residual_accepted':int(original.original_residual_acceptance.sum()),'clean_improved':int(original.clean_recovery_improved.sum()),
             'clean_harmed':int(original.clean_recovery_harmed.sum()),'accepted_but_harmed':len(bad),
             'clean_test_false_alarm_hours_across_models':int(clean_false_alarms.false_alarm_hours.sum()),
             'clean_test_hours_across_models':int(clean_false_alarms.hours.sum()),
             'new_live_llama_calls':0,'independent_sites':1,'independent_test_weeks':2,
             'scope':'Exploratory repeated-model evaluation, not 168 independent trajectories or a physical-control validation.'}
    return by_attack,methods,summary


def implementation_checks():
    """Counterexamples to incorrect semantics, not a claim of model efficacy."""
    from .core import recognized_protocol
    from .agents import parse_output
    parsed,error=parse_output('not JSON','defense','current')
    x=np.ones((3,4),dtype=np.float32)
    # The current predictor changes after replacement, so residual gets worse.
    def adverse_predict(v):
        return np.zeros_like(v) if np.all(v==1) else np.full_like(v,10)
    result=replace_and_verify(x,adverse_predict,0,2,'DIGITAL_TWIN_OVERRIDE')
    assert not result['accepted'] and not np.array_equal(result['defended'],x)
    return {'substring_accepts_NO_OVERRIDE':recognized_protocol('NO_OVERRIDE'),
            'malformed_defense_falls_back_to_active_override':bool(error and parsed['command']=='DIGITAL_TWIN_OVERRIDE'),
            'failed_candidate_is_returned_without_rollback':not result['accepted'] and not np.array_equal(result['defended'],x),
            'scope':'Observed implementation semantics; the tutorial flags them and uses explicit policy for optional new execution.'}


def strict_protocol(record):
    """Fail closed for optional new Llama use; do not silently invoke a fallback."""
    if record.get('fallback_used') or record.get('parse_error'):
        return 'ABSTAIN'
    command=record.get('parsed',{}).get('command')
    return command if command in {'DIGITAL_TWIN_OVERRIDE','ASSET_ISOLATION'} else 'ABSTAIN'


def distribution_diagnostics(raw,scaled,indices,scaler,models,metadata,folder):
    """Post-evaluation explanation only; it does not refit or select any model."""
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    names=metadata['observation_names'];M=len(names);B=metadata['buildings']
    rows=[];summaries=[]
    for seed,rec in models.items():
        for split,z in scaled.items():
            error=(z-rec['predict'](z))**2
            per_feature=error.reshape(len(z),B,M).mean(axis=(0,1))
            total=float(error.mean())
            summaries.append({'seed':seed,'split':split,'clean_reconstruction_mse':total,
                              'month_fraction_of_squared_error':float(per_feature[names.index('month')]/per_feature.sum())})
            for j,name in enumerate(names):
                rows.append({'seed':seed,'split':split,'feature':name,'mean_squared_residual':float(per_feature[j]),
                             'fraction_of_squared_error':float(per_feature[j]/per_feature.sum())})
    pd.DataFrame(rows).to_csv(folder/'feature_error_decomposition.csv',index=False)
    pd.DataFrame(summaries).to_csv(folder/'clean_distribution_shift.csv',index=False)
    month=names.index('month');calendar=[]
    for split,ix in indices.items():
        values,counts=np.unique(raw[ix,month],return_counts=True)
        calendar.append({'split':split,'month_values':values.tolist(),'counts':counts.tolist(),
                         'training_scale_for_month':float(scaler.scale_[month]),
                         'standardized_month_min':float(scaled[split][:,month].min()),
                         'standardized_month_max':float(scaled[split][:,month].max())})
    save_json(folder/'calendar_shift.json',calendar)
    return pd.DataFrame(summaries),pd.DataFrame(calendar)
