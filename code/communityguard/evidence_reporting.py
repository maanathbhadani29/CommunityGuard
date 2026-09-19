"""Report measured outcomes and authentic Llama transcripts without rerunning inference.

This module requires only numpy/pandas and, for PDF compilation, pdflatex.
It never changes a diagnosis, repair, scientific model, or archived response.
The same functions are called by the full study runner and the tutorial.
"""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess

import numpy as np
import pandas as pd

MODES = ("llama_one_pass", "llama_langgraph")
ATTACKS = ("market_hack", "meter_hack", "inverter_hack", "time_spoofing")
LABELS = ("Price", "Load", "PV", "Clock")


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def esc(text):
    return str(text).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def summarize_evidence(run):
    """Recompute all supplementary summaries directly from case metrics and logs."""
    run = Path(run)
    out = run / "evidence"
    out.mkdir(exist_ok=True)
    d = pd.read_csv(run / "all_numerical_metrics.csv")
    g = d[(d.method == "guarded_rules") & d.attack.notna()].copy()
    frames = {}
    for name, field in [("severity", "severity"), ("cohort", "cohort"), ("channel", "attack")]:
        aggregations = dict(
            records=("case_id", "size"), improved=("improved", "sum"),
            unchanged=("unchanged", "sum"), harmed=("harmed", "sum"),
            mean_error_before=("error_before", "mean"), mean_error_after=("error_after", "mean"))
        # Native-unit RMSE is meaningful within one channel, never pooled across
        # price, energy and hour units in severity/cohort summaries.
        if name == "channel":
            aggregations.update(mean_target_rmse_before=("target_rmse_before", "mean"),
                                mean_target_rmse_after=("target_rmse_after", "mean"))
        frame = g.groupby(field).agg(**aggregations)
        frame["improved_percent"] = 100 * frame.improved / frame.records
        frame["mean_error_reduction_percent"] = 100 * (1-frame.mean_error_after/frame.mean_error_before)
        frame.to_csv(out / f"{name}_results.csv")
        frames[name] = frame
    g[g.harmed].to_csv(out / "harmful_cases.csv", index=False)
    rows = []
    for file in sorted(run.glob("phase_*/live_llama/*/logs/*/*.json")):
        r = json.loads(file.read_text())
        rel = file.relative_to(run)
        cohort, _, mode, _, case, filename = rel.parts
        if mode not in MODES:
            continue
        response = r.get("response", {})
        raw = r.get("raw", "")
        if response and raw != response.get("message", {}).get("content", ""):
            raise ValueError(f"Raw model text differs from server response: {rel}")
        if r.get("source") not in ("llama_live", "llama_archived_replay"):
            raise ValueError(f"Non-Llama record in model evidence: {rel}")
        rows.append(dict(cohort=cohort, method=mode, case_id=case, stage=r["stage"],
                         call=filename, input_tokens=r.get("input_tokens") or 0,
                         output_tokens=r.get("output_tokens") or 0,
                         cached_input_tokens=response.get("prompt_eval_cached_count", 0) or 0,
                         latency_seconds=r.get("latency_seconds", 0),
                         invalid=bool(r.get("error")), source=r["source"],
                         raw_response=raw, log_file=str(rel),
                         log_sha256=hashlib.sha256(file.read_bytes()).hexdigest()))
    calls = pd.DataFrame(rows)
    if calls.empty:
        raise ValueError("No actual Llama response records found.")
    if not ((calls.cached_input_tokens >= 0) & (calls.cached_input_tokens <= calls.input_tokens)).all():
        raise ValueError("Invalid prompt-cache accounting.")
    calls["uncached_input_tokens"] = calls.input_tokens-calls.cached_input_tokens
    calls.to_csv(out / "llama_calls.csv", index=False)
    for name, group in [("mode", ["method"]), ("stage", ["method", "stage"])]:
        frame = calls.groupby(group).agg(
            calls=("call", "size"), input_tokens=("input_tokens", "sum"),
            output_tokens=("output_tokens", "sum"), cached_input_tokens=("cached_input_tokens", "sum"),
            uncached_input_tokens=("uncached_input_tokens", "sum"),
            model_call_seconds=("latency_seconds", "sum"), invalid_calls=("invalid", "sum"))
        frame["cached_percent"] = 100 * frame.cached_input_tokens/frame.input_tokens
        frame.to_csv(out / f"llama_{name}_tokens.csv")
        frames[f"tokens_{name}"] = frame
    # The established summary is an independent cross-check of raw-log accounting.
    expected = pd.read_csv(run / "llama_costs.csv").set_index("method")
    for key in ["calls", "input_tokens", "output_tokens", "invalid_calls"]:
        np.testing.assert_array_equal(frames["tokens_mode"][key], expected.loc[frames["tokens_mode"].index, key])
    frames["calls"] = calls
    frames["guarded"] = g
    return frames


def read_exchange(run, cohort, case):
    root = Path(run) / cohort / "live_llama/llama_langgraph"
    logs = [json.loads(f.read_text()) for f in sorted((root / "logs" / case).glob("*.json"))]
    trace = json.loads((root / "states" / case / "graph_trace.json").read_text())
    return logs, trace


def write_fragments(run, out, frames):
    run, out = Path(run), Path(out)
    s = pd.read_csv(run / "numerical_summary.csv").set_index("method")
    paired = pd.read_csv(run / "paired_summary.csv").set_index("method")
    r = s.loc["guarded_rules"]
    clean_rules = pd.read_csv(run / "clean_summary.csv").set_index("method").loc["guarded_rules"]
    paired_data = pd.read_csv(run / "paired_metrics.csv")
    paired_clean = paired_data[paired_data.attack.isna()]
    reduction = 100 * (1-r.mean_error_after/r.mean_error_before)
    nums = dict(TrainH=4032, CalH=1344, TestH=2688, CaseN=int(r.cases),
                ImproveN=int(r.improved), HarmN=int(r.harmed), UnchangedN=int(r.unchanged),
                JointN=int(r.joint_location_channel), BeforeE=f"{r.mean_error_before:.6f}",
                AfterE=f"{r.mean_error_after:.6f}", ErrorReduction=f"{reduction:.1f}",
                ImprovePercent=f"{100*r.improved/r.cases:.1f}", HarmPercent=f"{100*r.harmed/r.cases:.2f}",
                TypedHarm=int(s.loc["typed_ae_only", "harmed"]), OriginalHarm=int(s.loc["original_block", "harmed"]),
                LlamaImprove=int(paired.loc["llama_langgraph", "improved"]),
                OneImprove=int(paired.loc["llama_one_pass", "improved"]),
                PairedRulesImprove=int(paired.loc["guarded_rules", "improved"]))
    nums.update(CleanN=int(clean_rules.records), CleanHarmN=int(clean_rules.harmed),
                PairedCaseN=int(paired.loc["guarded_rules", "cases"]),
                PairedCleanN=int((paired_clean.method == "guarded_rules").sum()),
                LlamaHarmN=int(paired.loc["llama_langgraph", "harmed"]),
                OneHarmN=int(paired.loc["llama_one_pass", "harmed"]),
                PairedRulesHarmN=int(paired.loc["guarded_rules", "harmed"]))
    nums["CleanAbstract"] = ("No clean control was harmed." if nums["CleanHarmN"] == 0 else
                             f"It harmed {nums['CleanHarmN']} of {nums['CleanN']} clean controls.")
    if nums["OneImprove"] == nums["LlamaImprove"] and nums["OneHarmN"] == nums["LlamaHarmN"]:
        nums["PairedAbstract"] = (f"Both Llama configurations improved {nums['LlamaImprove']} of {nums['PairedCaseN']} attacked records, "
                                  f"compared with {nums['PairedRulesImprove']} for rules.")
        if nums["LlamaHarmN"] == 0:
            nums["PairedAbstract"] += " Neither Llama configuration harmed a subset record."
        else:
            nums["PairedAbstract"] += f" Each harmed {nums['LlamaHarmN']} subset records."
    else:
        nums["PairedAbstract"] = (f"One-pass and full-graph Llama improved {nums['OneImprove']} and {nums['LlamaImprove']} "
                                  f"of {nums['PairedCaseN']} attacks, versus {nums['PairedRulesImprove']} with rules; "
                                  f"they harmed {nums['OneHarmN']} and {nums['LlamaHarmN']} records, respectively.")
    (out / "update_numbers.tex").write_text("\n".join("\\newcommand{\\"+k+"}{"+str(v)+"}" for k,v in nums.items())+"\n")
    save_json(out / "reported_values.json", nums)
    labels = {"no_intervention":"No intervention", "original_block":"Full-building AE", "typed_ae_only":"Feature-aware AE", "guarded_rules":"CommunityGuard (rules)"}
    table = [r"\begin{table}[t]", r"\caption{Telemetry recovery on all \CaseN{} attacked records}", r"\label{tab:updated}", r"\centering\footnotesize", r"\begin{tabular}{lrrrr}", r"\toprule", r"Method & Better & Same & Worse & Mean $E$ \\", r"\midrule"]
    for method, label in labels.items():
        v = s.loc[method]
        table.append(f"{label} & {int(v.improved)} & {int(v.unchanged)} & {int(v.harmed)} & {v.mean_error_after:.6f} \\\\")
    table += [r"\bottomrule", r"\end{tabular}", r"\vspace{2pt}", r"\parbox{\columnwidth}{\scriptsize Better/same/worse compare repaired and received-input errors. Lower $E$ is better. CommunityGuard uses rule-based diagnosis and planning here; the Llama comparison uses a separate paired subset.}", r"\end{table}"]
    (out / "update_table.tex").write_text("\n".join(table)+"\n")
    c = frames["channel"]
    details = [f"{label.lower()} {int(c.loc[a].improved)}/{int(c.loc[a].unchanged)}/{int(c.loc[a].harmed)}" for a,label in zip(ATTACKS,LABELS)]
    text = f"Of {int(c.records.iloc[0])} attacked records per channel, improved/unchanged/harmed counts are " + "; ".join(details) + ". "
    text += (f"Mean per-record target RMSE decreases from {c.loc['meter_hack'].mean_target_rmse_before:.3f} to {c.loc['meter_hack'].mean_target_rmse_after:.3f}~kWh for load and "
             f"from {c.loc['inverter_hack'].mean_target_rmse_before:.3f} to {c.loc['inverter_hack'].mean_target_rmse_after:.3f}~kWh for PV. "
             "These native-unit errors are averaged over all records of the corresponding attack, including abstentions. ")
    clean = pd.read_csv(run / "clean_summary.csv").set_index("method").loc["guarded_rules"]
    text += f"CommunityGuard-AI with rules raises {int(clean.alarms)} alarms and harms {int(clean.harmed)} of {int(clean.records)} clean controls.\n"
    (out / "channel_results.tex").write_text(text)
    details = []
    for key,label in [("mild","mild"),("original","strong under-reporting"),("overreport","over-reporting")]:
        v = frames["severity"].loc[key]
        details.append(f"{label}: {int(v.improved)}/{int(v.unchanged)}/{int(v.harmed)}")
    t = f"Across {int(frames['severity'].records.iloc[0])} records per setting, improved/unchanged/harmed counts are " + "; ".join(details) + ". "
    severity = frames["severity"]
    if severity.loc["mild", "unchanged"] > severity.drop("mild").unchanged.max():
        t += "The higher unchanged count for mild attacks indicates reduced intervention at lower perturbation magnitudes. "
    a,b = (frames["cohort"].loc[k] for k in ["phase_1","phase_2"])
    t += f"The two cohorts yield {int(a.improved)}/{int(a.records)} and {int(b.improved)}/{int(b.records)} improved records, with {int(a.harmed)} and {int(b.harmed)} harmed records, respectively. "
    harmful = frames["guarded"][frames["guarded"].harmed]
    opposite = ((harmful.attack.eq("meter_hack") & harmful.selected_channel.eq("solar_generation")) | (harmful.attack.eq("inverter_hack") & harmful.selected_channel.eq("non_shiftable_load")))
    if len(harmful) and opposite.all() and harmful.localized_correctly.all():
        t += f"All {len(harmful)} harmful outputs select the correct building but confuse load and PV, editing the initially healthy complementary channel. This isolates channel attribution, rather than building selection, as the observed harmful failure mechanism."
    (out / "stratified_results.tex").write_text(t+"\n")
    t = []
    for method,label in [("guarded_rules","Rules"),("llama_one_pass","One-pass Llama"),("llama_langgraph","Full-graph Llama")]:
        v = paired.loc[method]
        t.append(f"{label}: {int(v.improved)} improved, {int(v.unchanged)} unchanged, {int(v.harmed)} harmed; correct building/channel in {int(v.joint_location_channel)}/{int(v.cases)}.")
    lm = pd.read_csv(run / "all_llama_metrics.csv")
    clean = lm[lm.attack.isna()]
    costs = pd.read_csv(run / "llama_costs.csv").set_index("method")
    t.append(f"The two Llama modes harm {int(clean.harmed.sum())} of their {len(clean)} clean-control executions. Across {int(costs.calls.sum())} actual model calls, {int(costs.invalid_calls.sum())} fail parsing or transport. Full-graph Llama requests {int(costs.loc['llama_langgraph','inspections'])} peer inspections and performs {int(costs.loc['llama_langgraph','replans'])} replanning attempt.")
    if all(abs(int(paired.loc[m, "improved"])-int(paired.loc["guarded_rules", "improved"])) <= 1
           for m in MODES) and all(paired.loc[m, "harmed"] == paired.loc["guarded_rules", "harmed"] for m in MODES):
        t.append("Llama closely matches the rules configuration on this subset, while supplying language-based diagnosis and tool selection within the shared workflow.")
    else:
        t.append("These paired counts measure the effect of the decision policy within the shared tools and guard.")
    (out / "llama_results.tex").write_text(" ".join(t)+"\n")
    modes = frames["tokens_mode"]
    t = []
    for mode,label in [("llama_one_pass","One-pass Llama"),("llama_langgraph","Full-graph Llama")]:
        v = modes.loc[mode]
        t.append(f"{label} uses {int(v.calls)} calls, {int(v.input_tokens):,} input tokens, and {int(v.output_tokens):,} output tokens; {int(v.cached_input_tokens):,} input tokens are served from cache ({v.cached_percent:.1f}\\%).")
    t.append("Counts are taken from the server's prompt, response, and cached-prompt fields \\cite{ollamaapi}; cached tokens are a subset of input tokens, not an additional workload.")
    t.append(f"Median summed model-call time per record, including clean controls, is {costs.loc['llama_one_pass','median_record_seconds']:.1f}~s and {costs.loc['llama_langgraph','median_record_seconds']:.1f}~s, respectively. One-pass always precedes full-graph inference for each case, so caching and shared CPU execution confound a latency comparison. These are model-call measurements, not end-to-end control deadlines.")
    (out / "token_results.tex").write_text(" ".join(t)+"\n")
    logs, trace = read_exchange(run, "phase_1", "original_meter_hack_b3_w1")
    snippets=[]
    for r in logs:
        label = "Forensic response" if r["stage"] == "forensics" else "Planning response"
        snippets.append(r"\noindent\begin{minipage}{\columnwidth}")
        snippets.append(r"\noindent\textit{"+label+f"; {r['input_tokens']} input / {r['output_tokens']} output tokens"+r".}\par")
        snippets += [r"\begin{lstlisting}[basicstyle=\ttfamily\scriptsize,aboveskip=3pt,belowskip=4pt]", json.dumps(json.loads(r["raw"]), indent=1), r"\end{lstlisting}"]
        snippets.append(r"\end{minipage}\par\smallskip")
    (out / "llama_excerpt.tex").write_text("\n".join(snippets)+"\n")
    stress = pd.concat([pd.read_csv(run / c / "assumption_stress.csv") for c in ["phase_1","phase_2"]])
    (out / "stress_results.tex").write_text(f"A separate {len(stress)}-record stress experiment corrupts load and net import together while preserving their mutual balance. The guarded engine improves {int(stress.improved.sum())}, leaves {int(stress.unchanged.sum())} unchanged, and harms {int(stress.harmed.sum())}. This common-mode failure exposes the dependence on an independently protected measurement; these cases are excluded from the single-channel benchmark.\n")
    failure_logs, failure_trace = read_exchange(run,"phase_2","original_meter_hack_b4_w1")
    save_json(run / "evidence" / "selected_exchanges.json", {
        "successful_load_repair":{"cohort":"phase_1","case_id":"original_meter_hack_b3_w1","calls":logs,"graph":trace},
        "rejected_retry":{"cohort":"phase_2","case_id":"original_meter_hack_b4_w1","calls":failure_logs,"graph":failure_trace}})
    return nums


def refresh_paper(code, run, compile_pdf=True):
    """Rebuild reporting from completed results; no retraining or new model calls."""
    code, run = Path(code).resolve(), Path(run).resolve()
    out = run / "paper"
    out.mkdir(exist_ok=True)
    for name in ["CommunityGuard_AI_Updated.tex", "IEEEtran.cls"]:
        shutil.copy2(code.parent / "manuscript" / name, out / name)
    from .case_reporting import plot_cases, write_case_captions
    from .update_plots import workflow
    workflow(run / "figures")
    plot_cases(code, run)
    shutil.copytree(run / "figures", out / "figures", dirs_exist_ok=True)
    from .final_study import summarize_study
    summarize_study(run)
    frames = summarize_evidence(run)
    write_fragments(run, out, frames)
    write_case_captions(code, run, out)
    from .comparison_reporting import write_comparison_table
    write_comparison_table(code, run, out)
    from .appendix_reporting import write_appendix
    write_appendix(run, out)
    if compile_pdf:
        if not shutil.which("pdflatex"):
            raise RuntimeError("Install pdflatex or pass compile_pdf=False.")
        for i in range(2):
            result = subprocess.run(["pdflatex","-interaction=nonstopmode","-halt-on-error","CommunityGuard_AI_Updated.tex"],cwd=out,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT, timeout=120)
            (out / f"compile_pass_{i+1}.txt").write_text(result.stdout)
            if result.returncode:
                raise RuntimeError(f"LaTeX failed; inspect {out / f'compile_pass_{i+1}.txt'}")
    return out
