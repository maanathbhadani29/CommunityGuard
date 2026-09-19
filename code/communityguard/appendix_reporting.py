"""Connect authentic prompt evidence, model text, and independently scored output."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .evidence_reporting import read_exchange


def prompt_evidence(record):
    prompt = record['request']['messages'][0]['content']
    return json.loads(prompt.split('EVIDENCE:\n', 1)[1].split('\nSCHEMA:', 1)[0])


def write_appendix(run, paper):
    run, paper = Path(run), Path(paper)
    case = 'original_meter_hack_b3_w1'
    logs, trace = read_exchange(run, 'phase_1', case)
    forensic = next(r for r in logs if r['stage'] == 'forensics')
    planning = next(r for r in logs if r['stage'] == 'planning')
    evidence = prompt_evidence(forensic)
    options = prompt_evidence(planning)['operations']
    candidates = evidence['candidates']
    selected = next(c for c in candidates if c['building'] == 3 and c['channel'] == 'non_shiftable_load')
    eligible = [o for o in options if o['accepted']]
    assert [o['operation'] for o in eligible] == ['ENERGY_BALANCE']
    assert trace['proposal']['operation'] == 'ENERGY_BALANCE'
    metrics = pd.read_csv(run/'all_llama_metrics.csv')
    metric = metrics[(metrics.cohort == 'phase_1') & (metrics.case_id == case) & (metrics.method == 'llama_langgraph')].iloc[0]
    # Case reporting has already verified unselected coordinates against saved arrays.
    figure = pd.read_csv(run/'evidence/figure_cases.csv')
    details = figure[figure.attack == 'meter_hack'].iloc[0]
    assert details.other_coordinates_unchanged
    changed = int(trace['verification']['changed_hours'])
    assert changed == int(metric.changed_entries) == int(details.edited_samples)
    failcase = 'original_meter_hack_b4_w1'
    failure, failed_trace = read_exchange(run, 'phase_2', failcase)
    failed_arrays = np.load(run/'phase_2/live_llama/llama_langgraph/states'/failcase/'response.npz')
    np.testing.assert_array_equal(failed_arrays['input'], failed_arrays['output'])
    assert failed_trace['attempts'] == 2 and failed_trace['diagnosis']['building'] == 1
    assert [json.loads(r['raw'])['operation'] for r in failure if r['stage'] == 'planning'] == ['ENERGY_BALANCE', 'PEER_CONSENSUS']
    text = (
        r'\textit{Evidence supplied.} For the load attack in Fig.~\ref{fig:load}, the forensic prompt asks for the best-supported building/channel. '
        f"It reports that {100*selected['inconsistent_fraction']:.3f}\\% of Building~3's load samples are inconsistent, above the {100*evidence['alarm_fraction']:g}\\% alarm threshold. "
        f"The planning prompt lists four tools; only energy balance has eligible samples ({changed} hours). "
        'The complete received responses appear below; only whitespace was changed. Token counts include the full prompts and schemas.\n'
        r'\input{llama_excerpt.tex}'+'\n'
        r'\textit{Executed result.} LangGraph applies the selected tool; the guard edits '
        f'{changed} of 168 load samples and leaves all other channels unchanged. Load RMSE falls from {metric.target_rmse_before:.4f} to {metric.target_rmse_after:.4f}~kWh. '
        'The response value 0.1327 is a mean conditional lower bound on absolute-error reduction over accepted samples, not the measured RMSE reduction.\n\n'
        r'\textit{Failure evidence.} In cohort~2, Llama instead selects Building~1 during a Building~4 load attack. '
        'Both energy-balance and peer-consensus proposals fail verification; LangGraph stops after two attempts and retains every input value. '
        'Both full exchanges are in the tutorial. These cases document diagnosis, tool selection, verification, and fallback within the framework.\n'
    )
    (paper/'appendix_evidence.tex').write_text(text)
