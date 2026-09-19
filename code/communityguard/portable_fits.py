"""Restore the original active fitted parameters without version-specific pickles.

The protected-balance protocol never calls its fitted ExtraTrees validators.
Only the ridge/scaler/peer parameters actually used by that protocol are exported.
No estimator is refitted and no expected repaired value is used during execution.
"""
from pathlib import Path
import hashlib, json, shutil
import numpy as np
import pandas as pd
import torch
from .pipeline import build_model
from .update_method import RepairTools
from .study import save_json


class ArrayScaler:
    def __init__(self, mean, scale):
        self.mean_, self.scale_ = np.asarray(mean), np.asarray(scale)
    def transform(self, values):
        # Preserve StandardScaler's two in-place rounding operations on float32.
        result = np.array(values, copy=True)
        result -= self.mean_
        result /= self.scale_
        return result
    def inverse_transform(self, values):
        result = np.array(values, copy=True)
        result *= self.scale_
        result += self.mean_
        return result


class ArrayRidge:
    def __init__(self, coef, intercept):
        self.coef_, self.intercept_ = np.asarray(coef), np.asarray(intercept)
    def predict(self, values):
        return np.asarray(values) @ self.coef_.T + self.intercept_


def read_parameters(path):
    """Read editable JSON values, preserving the original numeric precision."""
    records = json.loads(Path(path).read_text())
    return {name:np.asarray(record['values'], dtype=record['dtype'])
            for name,record in records.items()}


def prepare_csv_refit(code, run, cohort, recollect=False):
    """Train a new experiment from the same editable CSV used in saved-fit mode."""
    if recollect:
        raise ValueError('New CityLearn simulation is an optional developer workflow. This tutorial reads the included CSV observations.')
    from .update_experiment import prepare
    code, run = Path(code), Path(run)/cohort
    data = code/'teaching_data'/cohort
    metadata = json.loads((data/'data_metadata.json').read_text())
    frame = pd.read_csv(data/'observations.csv', float_precision='round_trip')
    columns = [f'B{b+1}.{name}' for b in range(metadata['buildings']) for name in metadata['observation_names']]
    if frame.columns.tolist() != columns:
        raise ValueError('Observation columns must match column_dictionary.csv.')
    inputs = run/'00_csv_inputs'; inputs.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(inputs/'clean_observations.npz', raw=frame.to_numpy(dtype=np.float32))
    save_json(inputs/'data_metadata.json',metadata)
    return prepare(code,run,protocol_path=code/'protocols'/f'{cohort}.json',data_path=inputs)


def prepare_portable_cohort(code, run, cohort, archive=None):
    code, run = Path(code), Path(run)/cohort
    archive = Path(archive) if archive is not None else code/'paper_run'
    run.mkdir(parents=True,exist_ok=True)
    cfg = json.loads((code/'protocols'/f'{cohort}.json').read_text())
    assert cfg == json.loads((archive/cohort/'protocol.json').read_text())
    if cfg.get('reference_mode') != 'protected_balance':
        raise ValueError('Portable fits are exported only for the paper protected-balance protocol.')
    source = archive/cohort/'portable_models'
    meta = json.loads((code/'teaching_data'/cohort/'data_metadata.json').read_text())
    # The teaching workflow reads ordinary CSV, with one named column per
    # building/observation. Restore the original float32 acquisition precision.
    data_file = code/'teaching_data'/cohort/'observations.csv'
    expected_columns = [f'B{b+1}.{name}' for b in range(meta['buildings'])
                        for name in meta['observation_names']]
    frame = pd.read_csv(data_file, float_precision='round_trip')
    if frame.columns.tolist() != expected_columns:
        raise ValueError(f'Unexpected observation columns in {data_file}')
    raw = frame.to_numpy(dtype=np.float32)
    n = cfg['train_hours']; c = n+cfg['calibration_hours']
    raw = raw[:c+cfg['test_hours']]
    train, calibration, test = raw[:n], raw[n:c], raw[c:]
    tools = RepairTools(meta['observation_names'],meta['buildings'],cfg)
    a = read_parameters(source/'calibration.json')
    tools.scales, tools.delta = a['scales'], a['delta']
    tools.ae_losses = a['ae_losses'].tolist()
    tools.models, tools.context_scalers, tools.validators = {}, {}, {}
    tools.pv_peers, tools.pv_slopes = {}, {}
    a = read_parameters(source/'active_predictors.json')
    for b in range(tools.B):
        tools.pv_peers[b] = a[f'peers_{b}'].tolist()
        tools.pv_slopes[b] = a[f'slopes_{b}']
        for k in range(3):
            key = f'{b}_{k}'
            tools.context_scalers[b,k] = ArrayScaler(a[f'mean_{key}'],a[f'scale_{key}'])
            tools.models[b,k] = ArrayRidge(a[f'coef_{key}'],a[f'intercept_{key}'])
    torch.use_deterministic_algorithms(True)
    tools.ae_model, tools.ae_predict, _, _ = build_model(
        tools.ae_features(train[:1]),0,cfg['ae_learning_rate'],cfg['threads'])
    tools.ae_model.load_state_dict(torch.load(source/'feature_aware_ae.pt',map_location='cpu',weights_only=True))
    a = read_parameters(source/'legacy_scaler.json')
    scaler = ArrayScaler(a['mean'],a['scale'])
    model, predict, _, _ = build_model(scaler.transform(train[:1]).astype(np.float32),0,cfg['ae_learning_rate'],cfg['threads'])
    model.load_state_dict(torch.load(source/'legacy_ae.pt',map_location='cpu',weights_only=True))
    # This normalization is a training-derived preprocessing parameter, just
    # like the restored model scalers. Retain the original fitted value because
    # NumPy 2 changed float32 percentile interpolation at the last few digits.
    metric_record = json.loads((source/'metric_scale.json').read_text())['metric_scale']
    metric_scale = np.asarray(metric_record['values'],dtype=metric_record['dtype'])
    shutil.copytree(source,run/'01_models')
    save_json(run/'protocol.json',cfg)
    save_json(run/'model_provenance.json',{
        'mode':'saved_fits','refitted':False,'representation':'portable active parameters',
        'restored_stage':'original training and calibration',
        'omitted_unused_models':'ExtraTrees validators are never called by protected_balance references',
        'recomputed_stages':['attacks','numerical decisions','graph execution','repairs','evaluation','plots'],
        'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(source.iterdir()) if p.is_file()}})
    return {'code':code,'run':run,'config':cfg,'raw':raw,'metadata':meta,
            'train':train,'calibration':calibration,'test':test,'tools':tools,
            'legacy_scaler':scaler,'legacy_predict':predict,'metric_scale':metric_scale}
