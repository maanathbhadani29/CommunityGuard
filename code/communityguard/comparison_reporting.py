"""Matched case table from measured outcomes and genuine Llama call records."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .case_reporting import case_details

METHODS = ["guarded_rules", "llama_one_pass", "llama_langgraph"]


def matched_cases(code, run):
    """Expose exact per-case method comparisons; missing rules timing stays missing."""
    code, run = Path(code).resolve(), Path(run).resolve()
    cases, _, _, settings = case_details(code, run)
    numerical = pd.read_csv(run / "all_numerical_metrics.csv")
    language = pd.read_csv(run / "all_llama_metrics.csv")
    rows = []
    for _, case in cases.iterrows():
        truth = np.load(run / "phase_1" / f"example_{case.attack}.npz")["truth"]
        for method in METHODS:
            frame = numerical if method == "guarded_rules" else language
            selected = frame[(frame.cohort == case.cohort) & (frame.case_id == case.case_id) & (frame.method == method)]
            if len(selected) != 1:
                raise ValueError(f"Expected one paired result: {case.case_id}, {method}")
            metric = selected.iloc[0]
            state = (run / "phase_1/03_graph_states" / case.case_id if method == "guarded_rules"
                     else run / "phase_1/live_llama" / method / "states" / case.case_id)
            trace = json.loads((state / "graph_trace.json").read_text())
            arrays = np.load(state / "response.npz")
            # Recompute native target RMSE using the exact saved output.
            meta = json.loads((code / "teaching_data/phase_1/data_metadata.json").read_text())
            column = 2*meta["features_per_building"]+meta["observation_names"].index(case.channel)
            error = arrays["output"][:, column].astype(float)-truth[:, column].astype(float)
            if case.attack == "time_spoofing":
                error = (error+12) % 24-12
            np.testing.assert_allclose(np.sqrt(np.mean(error**2)), metric.target_rmse_after, rtol=1e-9, atol=1e-12)
            record = dict(figure=int(case.figure), attack=case.attack, cohort=case.cohort,
                          case_id=case.case_id, building=3, method=method,
                          operation=trace["proposal"]["operation"], edited_samples=int(metric.changed_entries),
                          perturbed_samples=int(case.corrupted_samples),
                          target_rmse_before=float(metric.target_rmse_before),
                          target_rmse_after=float(metric.target_rmse_after),
                          input_tokens=0, output_tokens=0, calls=0,
                          forensic_seconds=None, planning_seconds=None, live_graph_seconds=None,
                          latency_status="not recorded; no Llama calls")
            if method != "guarded_rules":
                logs = [json.loads(f.read_text()) for f in sorted((run / "phase_1/live_llama" / method / "logs" / case.case_id).glob("*.json"))]
                forensic = sum(v["latency_seconds"] for v in logs if v["stage"] == "forensics")
                planning = sum(v["latency_seconds"] for v in logs if v["stage"] == "planning")
                timing = metric
                if metric.decision_source == "llama_archived_replay":
                    archive = code / "paper_run"
                    # Historical timing is distinct from the newly computed repair.
                    saved = pd.read_csv(archive / "all_llama_metrics.csv")
                    timing = saved[(saved.cohort == case.cohort) & (saved.case_id == case.case_id) & (saved.method == method)].iloc[0]
                if timing.decision_source != "llama_live":
                    raise ValueError("Original live timing is required for the paper table.")
                record.update(input_tokens=sum(v["input_tokens"] for v in logs),
                              output_tokens=sum(v["output_tokens"] for v in logs), calls=len(logs),
                              forensic_seconds=forensic, planning_seconds=planning,
                              live_graph_seconds=float(timing.graph_wall_seconds),
                              latency_status="recorded original live execution")
                np.testing.assert_allclose(forensic+planning, timing.model_seconds, rtol=1e-10)
                assert record["input_tokens"] == int(metric.input_tokens)
                assert record["output_tokens"] == int(metric.output_tokens)
            rows.append(record)
    result = pd.DataFrame(rows)
    result.to_csv(run / "evidence/figure_baseline_comparison.csv", index=False)
    return result, settings


def write_comparison_table(code, run, paper):
    """A single full-width table compares attacks, decisions, outcomes and workload."""
    frame, p = matched_cases(code, run)
    labels = {"guarded_rules":"Guarded rules", "llama_one_pass":"One-pass Llama", "llama_langgraph":"Full-graph Llama"}
    tool = {"PEER_CONSENSUS":"Peer", "CONDITIONAL_MODEL":"Cond.", "ENERGY_BALANCE":"Balance", "AE_RECONSTRUCTION":"AE"}
    attacks = [
        f"Price (Fig.~3): $\\widetilde C={p['market_multiplier']:g}C+\\epsilon$, noise SD {p['market_noise_sd']:g}",
        f"Load (Fig.~4): $\\widetilde\\ell={p['meter_multiplier']:g}\\ell$",
        f"PV (Fig.~5): $\\widetilde p=u_t p$, $u_t\\sim\\mathcal U({p['inverter_uniform'][0]:g},{p['inverter_uniform'][1]:g})$",
        f"Clock (Fig.~6): {p['clock_shift_hours']:g}-hour delay modulo 24"]
    units = ["currency/kWh", "kWh", "kWh", "h"]
    lines = [r"\begin{table*}[!t]",
             r"\caption{CommunityGuard-AI decision policies on matched attacks: Building 3, cohort 1, week 1}",
             r"\label{tab:case-comparison}", r"\centering\footnotesize",
             r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llrrrrr@{}}", r"\toprule",
             r"Method & Tool & Edited / perturbed & Target RMSE & Tokens in / out & Forensic / planning (s) & Graph (s) \\",
             r"\midrule"]
    for k, (_, group) in enumerate(frame.groupby("figure", sort=True)):
        before = group.iloc[0].target_rmse_before
        precision = 6 if k == 0 else 0 if k == 3 else 4
        description = attacks[k]+f"; received RMSE = {before:.{precision}f} {units[k]}"
        lines.append(r"\multicolumn{7}{@{}l}{\textit{"+description+r"}} \\")
        for _, row in group.iterrows():
            ftime = "--" if pd.isna(row.forensic_seconds) else f"{row.forensic_seconds:.2f} / {row.planning_seconds:.2f}"
            gtime = "NR" if pd.isna(row.live_graph_seconds) else f"{row.live_graph_seconds:.2f}"
            rmse = f"{row.target_rmse_after:.{precision}f}"
            lines.append(f"{labels[row.method]} & {tool[row.operation]} & {row.edited_samples} / {row.perturbed_samples} & {rmse} & {row.input_tokens:,} / {row.output_tokens:,} & {ftime} & {gtime} \\\\")
        lines.append(r"\addlinespace[3pt]" if k < 3 else r"\bottomrule")
    lines += [r"\end{tabular*}", r"\par\vspace{4pt}",
              r"\parbox{\textwidth}{\scriptsize All three policies share the framework's tools and guard. Attacks span 168 hourly samples; figures show the first 72. RMSE uses all 168 samples and the units stated above. Peer = peer consensus; Cond. = conditional model (peer-hour median for clock); Balance = energy balance. Tokens sum both Llama calls and include cached input tokens. -- = no Llama calls; NR = rule-based wall-clock latency was not recorded. One-pass ran before full-graph, so caching confounds timing comparisons. Graph time excludes collection/training/scoring. Replay retains original live timings.}",
              r"\end{table*}"]
    (Path(paper) / "case_comparison_table.tex").write_text("\n".join(lines)+"\n")
    return frame
