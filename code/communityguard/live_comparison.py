"""Compare authentic replay with fresh Ollama inference without requiring agreement.

The original scientific modules are reused unchanged. This module handles live
transport, progress, checkpoints, comparisons, and plots. Test clients cannot be
passed to the public live runner. No archived decision is used as a live fallback.
"""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import time
import numpy as np
import pandas as pd
import requests

from .final_study import COHORTS, MODES, ReplayClient, selected_cases
from .portable_fits import prepare_portable_cohort, prepare_csv_refit
from .update_agents import LlamaDecisionClient, RulesClient
from .update_experiment import metrics
from .update_graph import build_defense_graph, run_graph
from .core import CHANNELS as ATTACK_CHANNELS

ATTACKS = ['market_hack', 'meter_hack', 'inverter_hack', 'time_spoofing']
KEYS = ['cohort', 'case_id', 'method']


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def array_hash(a):
    a = np.ascontiguousarray(a)
    return hashlib.sha256(str((a.shape, a.dtype.str)).encode() + a.tobytes()).hexdigest()


class LiveTransportError(RuntimeError):
    pass


class ProgressLiveClient(LlamaDecisionClient):
    """Original prompt/options/parser, with a direct bounded HTTP transport."""
    def __init__(self, url='http://127.0.0.1:11434', model='llama3.2:3b', seed=42, timeout=180):
        super().__init__(url=url, model=model, seed=seed, timeout=timeout)
        if timeout <= 0:
            raise ValueError('The request timeout must be positive.')
        self.session = requests.Session()
        self.session.trust_env = False

    def request(self, path, payload=None):
        # In particular, do not route through the historical file-queue bridge.
        timeout = (5, self.timeout if path == '/api/chat' else min(self.timeout, 15))
        response = self.session.request('GET' if payload is None else 'POST',
            self.url + path, json=payload, timeout=timeout)
        response.raise_for_status()
        value = response.json()
        if not isinstance(value, dict) or value.get('error'):
            raise LiveTransportError(f'Ollama returned an error: {value}')
        return value

    def preflight(self):
        try:
            manifest = super().preflight()
            manifest['runtime'] = self.request('/api/version')
        except Exception as exc:
            raise LiveTransportError(
                f'Cannot use {self.model} at {self.url}. Open Ollama, run '
                f'"ollama pull {self.model}" in a terminal, and rerun this cell. '
                f'No live experiment has started. Details: {exc}') from exc
        manifest['request_timeout_seconds'] = self.timeout
        manifest['transport'] = 'direct HTTP; no archive fallback'
        return manifest

    def decide(self, stage, evidence, path):
        print(f'    Fresh Ollama call: {stage} ({self.model})', flush=True)
        record = super().decide(stage, evidence, path)
        # The base client catches errors for graph abstention and records them.
        record['transport_ok'] = isinstance(record.get('response'), dict)
        write_json(path, record)
        print(f'    Returned in {record["latency_seconds"]:.2f}s; '
              f'tokens {record.get("input_tokens", "unknown")} / '
              f'{record.get("output_tokens", "unknown")}; '
              f'{"valid JSON decision" if not record.get("error") else record["error"]}', flush=True)
        return record


def open_comparison(code, folder, scope, live_fit_mode, client, model_manifest):
    """Open a new run, or explicitly resume a matching saved-fit live run."""
    code, folder = Path(code), Path(folder)
    if scope not in {'figures_only', 'paper_subset'}:
        raise ValueError('scope must be figures_only or paper_subset')
    if live_fit_mode not in {'saved', 'refit'}:
        raise ValueError('live_fit_mode must be saved or refit')
    files = sorted([*code.glob('communityguard/*.py'), *code.glob('protocols/*.json'),
                    *code.glob('teaching_data/phase_*/*.csv'),
                    *code.glob('teaching_data/phase_*/data_metadata.json'),
                    *code.glob('paper_run/phase_*/portable_models/*'),
                    *code.glob('paper_run/phase_*/live_llama/*/logs/*/*.json')])
    config = {'scope': scope, 'live_fit_mode': live_fit_mode, 'model': client.model,
              'seed': client.seed, 'url': client.url,
              'model_digest': model_manifest.get('installed_model', {}).get('digest'),
              'ollama_version': model_manifest.get('runtime', {}).get('version'),
              'files_sha256': {str(p.relative_to(code)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / 'comparison_manifest.json'
    if path.exists():
        manifest = json.loads(path.read_text(encoding='utf-8'))
        if manifest['configuration'] != config:
            raise ValueError('This run has different inputs, settings, model digest, or runtime. Start a new run folder.')
        if live_fit_mode == 'refit':
            raise ValueError('Start a new folder for refit mode; partial runs with retraining are not resumed.')
    else:
        manifest = {'configuration': config, 'created_utc': datetime.now(timezone.utc).isoformat(),
                    'status': 'created', 'live_completed': False,
                    'model_manifest': model_manifest,
                    'timing_note': 'Replay model/graph inference times are historical. Live times are measured now. Hardware, load and cache confound comparisons.'}
        write_json(path, manifest)
    return folder


def prepare_contexts(code, folder, scope='figures_only', live_fit_mode='saved'):
    """Shared inputs/metric scale; optional retraining changes only the live arm."""
    code, folder = Path(code), Path(folder)
    cohorts = ['phase_1'] if scope == 'figures_only' else COHORTS
    stage = folder / 'fit_stages' / datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    baseline, live = {}, {}
    for cohort in cohorts:
        baseline[cohort] = prepare_portable_cohort(code, stage/'saved', cohort)
        if live_fit_mode == 'refit':
            print(f'Training fresh models for {cohort} from the included CSV...', flush=True)
            live[cohort] = prepare_csv_refit(code, stage/'refit', cohort)
            np.testing.assert_array_equal(live[cohort]['test'], baseline[cohort]['test'])
            # The metric must have the same scale in both arms to be comparable.
            live[cohort]['metric_scale'] = baseline[cohort]['metric_scale'].copy()
        else:
            live[cohort] = baseline[cohort]
    cases = []
    for cohort in cohorts:
        for case in selected_cases(baseline[cohort], cohort):
            if scope == 'figures_only' and case['case_id'] not in ['clean_w1'] + [f'original_{a}_b3_w1' for a in ATTACKS]:
                continue
            cases.append((cohort, case))
    listing = [{'cohort': cohort, **{k: c[k] for k in ['case_id', 'attack', 'severity', 'window', 'building']},
                'input_sha256': array_hash(c['received'])} for cohort, c in cases]
    write_json(folder/'selected_cases.json', listing)
    return baseline, live, cases


def _case_folder(folder, arm, cohort, method):
    return Path(folder) / arm / cohort / method


def _token_sum(logs, key):
    values = [r.get(key) for r in logs]
    return None if any(v is None for v in values) else int(sum(values))


def execute_case(ctx, cohort, case, client, destination, method, attempts, inspection, arm):
    """Run one graph and then score it. No target outcome or truth enters the graph."""
    destination = Path(destination)
    case_id = case['case_id']
    result_file = destination / 'cases' / f'{case_id}.json'
    state_file = destination / 'states' / case_id / 'response.npz'
    if result_file.exists():
        row = json.loads(result_file.read_text(encoding='utf-8'))
        if row['transport_ok']:
            if row['input_sha256'] != array_hash(case['received']):
                raise ValueError('Saved case belongs to different input data.')
            if not state_file.exists() or hashlib.sha256(state_file.read_bytes()).hexdigest() != row['response_file_sha256']:
                raise ValueError('Saved case arrays are missing or changed. Start a new run folder.')
            print('    Already completed in this comparison; retained.', flush=True)
            return row
    # A failed case is retried as one new attempt, preserving its prior files.
    if result_file.exists() or (destination/'logs'/case_id).exists() or (destination/'states'/case_id).exists():
        suffix = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
        for old in [destination/'logs'/case_id, destination/'states'/case_id]:
            if old.exists():
                target = destination/'failed_attempts'/suffix/old.parent.name/case_id
                target.parent.mkdir(parents=True, exist_ok=True)
                old.rename(target)
        target = destination/'failed_attempts'/suffix/'case.json'
        target.parent.mkdir(parents=True, exist_ok=True)
        if result_file.exists():
            result_file.rename(target)
    started = time.perf_counter()
    graph = build_defense_graph(ctx['tools'], client, destination/'logs', attempts, inspection)
    result = run_graph(graph, case['received'], case_id, destination/'states')
    wall = time.perf_counter() - started
    logs = [json.loads(p.read_text(encoding='utf-8')) for p in sorted((destination/'logs'/case_id).glob('*.json'))]
    d = result.get('diagnosis', {})
    attack, b = case['attack'], case['building']
    row = {'arm': arm, 'cohort': cohort, 'method': method,
           **{k: case[k] for k in ['case_id', 'attack', 'window', 'severity']},
           'terminal': result['terminal'], 'selected_building': int(d.get('building', 0)),
           'selected_channel': d.get('channel', 'none'),
           'operation': result.get('proposal', {}).get('operation', 'NONE'),
           'attempts': int(result['attempts']), 'inspections': int(result['inspections']),
           'correct_target': bool(b is not None and d.get('building') == b+1 and d.get('channel') == ATTACK_CHANNELS[attack]),
           'calls': len(logs), 'invalid_calls': sum(bool(r.get('error')) for r in logs),
           'input_tokens': _token_sum(logs, 'input_tokens'),
           'output_tokens': _token_sum(logs, 'output_tokens'),
           'model_call_seconds': float(sum(r['latency_seconds'] for r in logs)),
           'current_graph_seconds': wall, 'inference_graph_seconds': wall if arm in {'live', 'rules'} else None,
           'source': client.source, 'transport_ok': all(r.get('transport_ok', True) for r in logs),
           'input_sha256': array_hash(case['received']),
           'response_file_sha256': hashlib.sha256(state_file.read_bytes()).hexdigest(),
           'state_file': str(state_file), 'timing_origin': 'historical live calls' if arm == 'replay' else 'current execution'}
    if arm == 'rules':
        # Rule decisions are logged but are not model calls or token use.
        row.update(calls=0, input_tokens=0, output_tokens=0, model_call_seconds=0.0)
    row.update(metrics(result['output'], case['received'], case['truth'], ctx, attack, b))
    np.savez_compressed(destination/'states'/case_id/'evaluation_truth.npz', truth=case['truth'])
    write_json(result_file, row)
    return row


def run_replay_and_rules(code, folder, contexts, cases):
    rows = []
    archive_timing = pd.read_csv(Path(code)/'paper_run/all_llama_metrics.csv').set_index(KEYS)
    for i, (cohort, case) in enumerate(cases, 1):
        for method, attempts, inspection in [('guarded_rules', 2, True), *MODES]:
            arm = 'rules' if method == 'guarded_rules' else 'replay'
            print(f'[{i}/{len(cases)}] {arm} | {cohort} | {method} | {case["case_id"]}', flush=True)
            client = RulesClient() if arm == 'rules' else ReplayClient(Path(code)/'paper_run'/cohort/'live_llama'/method/'logs')
            row = execute_case(contexts[cohort], cohort, case, client,
                _case_folder(folder, arm, cohort, method), method, attempts, inspection, arm)
            if arm == 'replay':
                row['inference_graph_seconds'] = float(archive_timing.loc[(cohort, case['case_id'], method), 'graph_wall_seconds'])
            rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(Path(folder)/'replay_and_rules_metrics.csv', index=False)
    return frame


def run_fresh_live(folder, contexts, cases, client):
    if type(client) is not ProgressLiveClient:
        raise TypeError('The live runner requires ProgressLiveClient and a real Ollama service.')
    folder = Path(folder)
    path = folder/'comparison_manifest.json'
    manifest = json.loads(path.read_text(encoding='utf-8'))
    manifest.update(status='live_running', live_completed=False)
    write_json(path, manifest)
    rows = []
    try:
        for i, (cohort, case) in enumerate(cases, 1):
            # Preserve the original one-pass-then-full-graph order; record its cache confound.
            for method, attempts, inspection in MODES:
                print(f'[{i}/{len(cases)}] LIVE | {cohort} | {method} | {case["case_id"]}', flush=True)
                row = execute_case(contexts[cohort], cohort, case, client,
                    _case_folder(folder, 'live', cohort, method), method, attempts, inspection, 'live')
                rows.append(row)
                pd.DataFrame(rows).to_csv(folder/'live_metrics_partial.csv', index=False)
                if not row['transport_ok']:
                    raise LiveTransportError('A live request failed. Its logs and abstained response were saved. '
                        'The batch stopped; the case was not accepted as a completed model evaluation. '
                        'Fix the service and rerun this cell to retry the failed case and retain completed cases.')
    except BaseException as exc:
        manifest.update(status='live_incomplete', live_completed=False, last_error=str(exc))
        write_json(path, manifest)
        raise
    frame = pd.DataFrame(rows)
    frame.to_csv(folder/'live_metrics.csv', index=False)
    manifest.update(status='live_complete', live_completed=True, completed_records=len(frame))
    manifest.pop('last_error', None)
    write_json(path, manifest)
    return frame


def outcome_summary(frame):
    rows = []
    for (arm, method), g in frame.groupby(['arm', 'method'], sort=False):
        a = g[g.attack.notna()]
        clean = g[g.attack.isna()]
        def total(field):
            return None if g[field].isna().any() else float(g[field].sum())
        rows.append({'arm': arm, 'method': method, 'attacks': len(a),
            'improved': int(a.improved.sum()), 'unchanged': int(a.unchanged.sum()), 'harmed': int(a.harmed.sum()),
            'correct_targets': int(a.correct_target.sum()), 'mean_error_before': a.error_before.mean(),
            'mean_error_after': a.error_after.mean(), 'mean_collateral_error': a.collateral_error.mean(),
            'clean_controls': len(clean), 'clean_harmed': int(clean.harmed.sum()),
            'clean_changed': int((clean.changed_entries > 0).sum()),
            'calls': int(g.calls.sum()), 'invalid_calls': int(g.invalid_calls.sum()),
            'input_tokens': total('input_tokens'), 'output_tokens': total('output_tokens'),
            'summed_model_seconds': total('model_call_seconds'),
            'median_inference_graph_seconds': g.inference_graph_seconds.median(),
            'median_current_graph_seconds': g.current_graph_seconds.median()})
    return pd.DataFrame(rows)


def compare_results(folder, baseline, live):
    folder = Path(folder)
    if live.empty or not live.transport_ok.all() or not live.source.eq('llama_live').all():
        raise ValueError('A complete genuine live run is required for a live comparison.')
    replay = baseline[baseline.arm == 'replay']
    if replay.duplicated(KEYS).any() or live.duplicated(KEYS).any():
        raise ValueError('Duplicate case keys in comparison.')
    joined = replay.merge(live, on=KEYS, suffixes=('_replay', '_live'), how='outer', validate='one_to_one', indicator=True)
    if not joined['_merge'].eq('both').all():
        raise ValueError('The live and replay case sets must match. A partial batch is not a full comparison.')
    deltas = []
    for _, r in joined.iterrows():
        if r.input_sha256_replay != r.input_sha256_live:
            raise ValueError('Live and replay inputs differ.')
        with np.load(r.state_file_replay) as a, np.load(r.state_file_live) as b:
            np.testing.assert_array_equal(a['input'], b['input'])
            changed = int(np.count_nonzero(np.abs(a['output'].astype(float)-b['output'].astype(float)) > 1e-7))
            identical = bool(np.array_equal(a['output'], b['output']))
        deltas.append({**{k: r[k] for k in KEYS}, 'attack': r.attack_replay,
            'same_target': bool(r.selected_building_replay == r.selected_building_live and r.selected_channel_replay == r.selected_channel_live),
            'same_operation': bool(r.operation_replay == r.operation_live),
            'same_terminal': bool(r.terminal_replay == r.terminal_live),
            'same_repair_array_exact': identical, 'different_entries_above_1e_minus7': changed,
            'error_replay': r.error_after_replay, 'error_live': r.error_after_live,
            'error_delta_live_minus_replay': r.error_after_live-r.error_after_replay,
            'target_rmse_replay': r.get('target_rmse_after_replay'),
            'target_rmse_live': r.get('target_rmse_after_live'),
            'collateral_error_replay': r.collateral_error_replay, 'collateral_error_live': r.collateral_error_live,
            'input_tokens_replay': r.input_tokens_replay, 'input_tokens_live': r.input_tokens_live,
            'output_tokens_replay': r.output_tokens_replay, 'output_tokens_live': r.output_tokens_live,
            'historical_model_seconds': r.model_call_seconds_replay, 'fresh_model_seconds': r.model_call_seconds_live})
    paired = pd.DataFrame(deltas)
    all_metrics = pd.concat([baseline, live], ignore_index=True)
    summary = outcome_summary(all_metrics)
    all_metrics.to_csv(folder/'all_comparison_metrics.csv', index=False)
    paired.to_csv(folder/'paired_differences.csv', index=False)
    summary.to_csv(folder/'comparison_summary.csv', index=False)
    channel_rows = []
    for (arm, method, attack), group in all_metrics[all_metrics.attack.notna()].groupby(['arm','method','attack']):
        channel_rows.append({'arm': arm, 'method': method, 'attack': attack, 'records': len(group),
            'mean_target_rmse': group.target_rmse_after.mean(), 'improved': int(group.improved.sum()),
            'harmed': int(group.harmed.sum())})
    pd.DataFrame(channel_rows).to_csv(folder/'channel_comparison.csv', index=False)
    report = {'status': 'PASS', 'meaning': 'Valid matched comparison; outcomes need not agree.',
        'case_method_pairs': len(paired), 'identical_repair_arrays': int(paired.same_repair_array_exact.sum()),
        'different_repair_arrays': int((~paired.same_repair_array_exact).sum()),
        'live_better_than_replay': int((paired.error_delta_live_minus_replay < -1e-12).sum()),
        'live_worse_than_replay': int((paired.error_delta_live_minus_replay > 1e-12).sum()),
        'error_unchanged': int((paired.error_delta_live_minus_replay.abs() <= 1e-12).sum()),
        'fresh_live_calls': int(live.calls.sum()), 'live_invalid_calls': int(live.invalid_calls.sum()),
        'no_equality_with_paper_required': True}
    write_json(folder/'comparison_check.json', report)
    return summary, paired, report


def read_calls(folder, arm, cohort, method, case_id):
    path = _case_folder(folder, arm, cohort, method)/'logs'/case_id
    return [json.loads(p.read_text(encoding='utf-8')) for p in sorted(path.glob('*.json'))]


def compare_calls(folder, cases):
    """Outer join accommodates new inspections, retries, missing calls and failures."""
    rows = []
    for cohort, case in cases:
        for method, _, _ in MODES:
            records = {}
            for arm in ['replay', 'live']:
                path = _case_folder(folder, arm, cohort, method)/'logs'/case['case_id']
                records[arm] = {p.name: json.loads(p.read_text(encoding='utf-8')) for p in path.glob('*.json')}
            for name in sorted(records['replay'].keys() | records['live'].keys()):
                a, b = records['replay'].get(name), records['live'].get(name)
                rows.append({'cohort': cohort, 'case_id': case['case_id'], 'method': method, 'call': name,
                    'replay_present': a is not None, 'live_present': b is not None,
                    'same_prompt': bool(a and b and a['request']['messages'] == b['request']['messages']),
                    'same_response_text': bool(a and b and a.get('raw') == b.get('raw')),
                    'same_parsed_decision': bool(a and b and a.get('parsed') == b.get('parsed')),
                    'input_tokens_replay': a.get('input_tokens') if a else None,
                    'input_tokens_live': b.get('input_tokens') if b else None,
                    'output_tokens_replay': a.get('output_tokens') if a else None,
                    'output_tokens_live': b.get('output_tokens') if b else None,
                    'live_error': b.get('error') if b else 'no corresponding live call'})
    frame = pd.DataFrame(rows)
    frame.to_csv(Path(folder)/'call_comparison.csv', index=False)
    return frame


def plot_comparison(folder, cases, paired, baseline, live):
    import matplotlib.pyplot as plt
    from .update_method import distance
    folder = Path(folder)
    images = folder/'comparison_figures'
    images.mkdir(exist_ok=True)
    combined = pd.concat([baseline, live])
    paths = []
    labels = ['Price', 'Load', 'PV', 'Clock']
    units = ['currency/kWh', 'kWh/step', 'kWh/step', 'hour']
    for k, attack in enumerate(ATTACKS):
        case_id = f'original_{attack}_b3_w1'
        cohort, case = next((co, c) for co, c in cases if co == 'phase_1' and c['case_id'] == case_id)
        rows = combined[(combined.cohort == cohort) & (combined.case_id == case_id) & (combined.method == 'llama_langgraph')].set_index('arm')
        a, b = np.load(rows.loc['replay','state_file']), np.load(rows.loc['live','state_file'])
        # The plotted coordinate is the injected target, even if diagnosis is wrong.
        # Observation ordering is declared in the unchanged input metadata.
        code = Path(json.loads((folder/'plot_context.json').read_text())['code'])
        names = json.loads((code/'teaching_data/phase_1/data_metadata.json').read_text())['observation_names']
        col = 2*len(names)+names.index(ATTACK_CHANNELS[attack])
        truth, received, replay, fresh = case['truth'][:,col], a['input'][:,col], a['output'][:,col], b['output'][:,col]
        n = min(72, len(truth)); h = np.arange(n)
        fig, axes = plt.subplots(2,1,figsize=(10,5.8),sharex=True,gridspec_kw={'height_ratios':[2,1]},constrained_layout=True)
        for values, color, style, label in [(truth,'#233b53','-','Clean'),(received,'#d1495b','--','Attacked'),
              (replay,'#007f73',':','Archived-response repair'),(fresh,'#7842a3','-.','Fresh Llama repair')]:
            axes[0].plot(h,values[:n],style,color=color,label=label,linewidth=1.8)
        axes[0].set_ylabel(f'{labels[k]} ({units[k]})')
        axes[0].legend(ncol=2,fontsize=9)
        for values,color,label in [(replay,'#007f73','Replay error'),(fresh,'#7842a3','Fresh error')]:
            axes[1].plot(h,np.abs(distance(values,truth,k))[:n],color=color,label=label)
        axes[1].set_ylabel('Absolute error');axes[1].set_xlabel('Hours since attack start (first 72 of 168)');axes[1].legend()
        r, l = rows.loc['replay'], rows.loc['live']
        fig.suptitle(f'{labels[k]} attack on B3 | Replay: B{r.selected_building}, {r.operation}\n'
                     f'Fresh: B{l.selected_building}, {l.operation} | Full-record collateral E: {l.collateral_error:.3g}',fontsize=11)
        for ax in axes: ax.grid(alpha=.2)
        path = images/f'{attack}_live_vs_replay.png'
        fig.savefig(path,dpi=160);plt.close(fig);paths.append(path)
        a.close();b.close()
    attacked = paired[paired.attack.notna()]
    fig, ax = plt.subplots(figsize=(10,3.8),constrained_layout=True)
    for method, g in attacked.groupby('method'):
        ax.plot(np.arange(len(g)),g.error_delta_live_minus_replay,marker='o',markersize=3,label=method)
    ax.axhline(0,color='black',linewidth=.8)
    ax.set_xlabel('Matched attacked record within each policy')
    ax.set_ylabel('Fresh error minus replay error')
    ax.set_title('Below zero: fresh run recovered better; above zero: replay recovered better')
    ax.legend();ax.grid(alpha=.2)
    path = images/'recovery_differences.png';fig.savefig(path,dpi=160);plt.close(fig);paths.append(path)
    return paths
