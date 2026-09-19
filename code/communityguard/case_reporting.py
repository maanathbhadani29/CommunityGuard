"""Explain and plot the four actual paper cases from saved arrays and live logs.

No scientific decisions are made here. Plotting truth is evaluation-only.
Recorded live latency remains separate from the duration of archival replay.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ATTACKS = ["market_hack", "meter_hack", "inverter_hack", "time_spoofing"]
CHANNELS = ["electricity_pricing", "non_shiftable_load", "solar_generation", "hour"]
LABELS = ["Price", "Load", "PV", "Clock"]
COLORS = {"truth": "#243b53", "attack": "#d1495b", "repair": "#007f73"}


def distance(x, truth, clock=False):
    delta = np.asarray(x, dtype=float)-np.asarray(truth, dtype=float)
    return (delta+12) % 24-12 if clock else delta


def case_details(code, run):
    """Return verified per-case data, including explicitly sourced live timings."""
    code, run = Path(code).resolve(), Path(run).resolve()
    metrics = pd.read_csv(run / "all_llama_metrics.csv")
    meta = json.loads((code / "teaching_data/phase_1/data_metadata.json").read_text())
    cfg = json.loads((run / "phase_1/protocol.json").read_text())
    settings = next(s for s in cfg["attack_severities"] if s["name"] == "original")
    names = meta["observation_names"]
    test_start = cfg["train_hours"]+cfg["calibration_hours"]
    rows, series, provenance = [], [], {}
    for k, attack in enumerate(ATTACKS):
        case = f"original_{attack}_b3_w1"
        row = metrics[(metrics.cohort == "phase_1") & (metrics.method == "llama_langgraph") & (metrics.case_id == case)].iloc[0]
        folder = run / "phase_1/live_llama/llama_langgraph"
        source = folder / "states" / case / "response.npz"
        example_file = run / "phase_1" / f"example_{attack}.npz"
        response, example = np.load(source), np.load(example_file)
        np.testing.assert_array_equal(response["input"], example["received"])
        trace = json.loads((source.parent / "graph_trace.json").read_text())
        logs = [json.loads(p.read_text()) for p in sorted((folder / "logs" / case).glob("*.json"))]
        col = 2*len(names)+names.index(CHANNELS[k])
        clean, received, repaired = example["truth"][:, col], response["input"][:, col], response["output"][:, col]
        pre, post = distance(received, clean, k == 3), distance(repaired, clean, k == 3)
        perturbed = np.abs(pre) > 1e-7
        edited = np.abs(repaired.astype(float)-received.astype(float)) > 1e-7
        retained = perturbed & ~edited
        other = np.arange(response["input"].shape[1]) != col
        np.testing.assert_array_equal(response["input"][:, other], response["output"][:, other])
        before, after = np.sqrt(np.mean(pre**2)), np.sqrt(np.mean(post**2))
        np.testing.assert_allclose([before, after], [row.target_rmse_before, row.target_rmse_after], rtol=1e-9, atol=1e-12)
        assert int(edited.sum()) == int(row.changed_entries)
        forensic = sum(r["latency_seconds"] for r in logs if r["stage"] == "forensics")
        planning = sum(r["latency_seconds"] for r in logs if r["stage"] == "planning")
        np.testing.assert_allclose(forensic+planning, row.model_seconds, rtol=1e-10)
        # Replay is quicker than live inference. Never relabel its measured wall
        # duration as the original live response latency.
        timing_row = row
        timing_source = str(run / "all_llama_metrics.csv")
        if row.decision_source == "llama_archived_replay":
            archive = code / "paper_run"
            # Only timing provenance is read from the original live execution.
            # Plot and recovery data always come from this run's response arrays.
            archived = pd.read_csv(archive / "all_llama_metrics.csv")
            timing_row = archived[(archived.cohort == "phase_1") & (archived.method == "llama_langgraph") & (archived.case_id == case)].iloc[0]
            timing_source = "paper_run/all_llama_metrics.csv (original live execution)"
            np.testing.assert_allclose(row.model_seconds, timing_row.model_seconds, rtol=1e-10)
        assert timing_row.decision_source == "llama_live", "Live response latency requires a live record."
        graph_seconds = float(timing_row.graph_wall_seconds)
        detail = dict(figure=k+3, cohort="phase_1", case_id=case, attacked_building=3,
                      detected_building=int(trace["diagnosis"]["building"]), channel=CHANNELS[k],
                      detected_channel=trace["diagnosis"]["channel"], attack=attack,
                      severity="original", week=1, attack_start_index=test_start,
                      attack_end_index=test_start+len(clean)-1, sampling_hours=1,
                      record_hours=len(clean), plotted_hours=72,
                      operation=trace["proposal"]["operation"], terminal=trace["terminal"],
                      corrupted_samples=int(perturbed.sum()), edited_samples=int(edited.sum()),
                      retained_corrupted_samples=int(retained.sum()),
                      target_rmse_before=float(before), target_rmse_after=float(after),
                      forensic_seconds=forensic, planning_seconds=planning,
                      recorded_live_graph_seconds=graph_seconds,
                      other_graph_seconds=graph_seconds-forensic-planning,
                      current_execution_graph_seconds=float(row.graph_wall_seconds),
                      timing_source=timing_source, decision_source=row.decision_source,
                      calls=len(logs), attempts=int(trace["attempts"]), inspections=int(trace["inspections"]),
                      input_tokens=sum(r["input_tokens"] for r in logs),
                      output_tokens=sum(r["output_tokens"] for r in logs),
                      other_coordinates_unchanged=True)
        rows.append(detail)
        series.append(pd.DataFrame(dict(case_id=case, figure=k+3, hour_since_attack_start=np.arange(len(clean)),
                                        trajectory_index=test_start+np.arange(len(clean)), clean=clean,
                                        attacked=received, defended=repaired, absolute_error_before=np.abs(pre),
                                        absolute_error_after=np.abs(post), perturbed=perturbed, edited=edited,
                                        retained_perturbed=retained)))
        provenance[attack] = dict(figure_number=k+3, case_id=case, cohort="phase_1", method="llama_langgraph",
                                  response_file=str(source.relative_to(run)), response_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                                  truth_file=str(example_file.relative_to(run)), plotted_hours=72, scored_hours=len(clean),
                                  target_column=col, selected_building=detail["detected_building"],
                                  terminal=detail["terminal"], operation=detail["operation"],
                                  attack_applied_hours=[0, len(clean)-1], trajectory_indices=[test_start, test_start+len(clean)-1],
                                  recorded_live_graph_seconds=graph_seconds,
                                  caption_metrics_source="evidence/figure_cases.csv")
    output = run / "evidence"
    output.mkdir(exist_ok=True)
    cases, data = pd.DataFrame(rows), pd.concat(series, ignore_index=True)
    cases.to_csv(output / "figure_cases.csv", index=False)
    data.to_csv(output / "figure_case_timeseries.csv", index=False)
    return cases, data, provenance, settings


def plot_cases(code, run):
    """Regenerate Figures 3-6 with target, attack, defense and latency labels."""
    run = Path(run)
    cases, data, provenance, p = case_details(code, run)
    out = run / "figures"
    out.mkdir(exist_ok=True)
    attacks = [f"Price x {p['market_multiplier']:g} + noise (SD {p['market_noise_sd']:g})",
               f"Load x {p['meter_multiplier']:g} ({100*(1-p['meter_multiplier']):g}% under-reporting)",
               f"PV x U({p['inverter_uniform'][0]:g}, {p['inverter_uniform'][1]:g})",
               f"Clock delayed {p['clock_shift_hours']:g} h; wrapped to 1-24"]
    defenses = {"CONDITIONAL_MODEL":"Conditional model", "ENERGY_BALANCE":"Energy balance",
                "PEER_CONSENSUS":"Peer consensus", "AE_RECONSTRUCTION":"Autoencoder"}
    plt.rcParams.update({"font.size":10, "axes.spines.top":False, "axes.spines.right":False, "pdf.fonttype":42})
    for k, row in cases.iterrows():
        d = data[(data.case_id == row.case_id) & (data.hour_since_attack_start < 72)]
        h = d.hour_since_attack_start.to_numpy()
        fig = plt.figure(figsize=(4.5, 3.6))
        gs = fig.add_gridspec(2, 1, height_ratios=[2, 1], left=.16, right=.985, bottom=.17, top=.76, hspace=.16)
        ax = fig.add_subplot(gs[0]); err = fig.add_subplot(gs[1], sharex=ax)
        fig.text(.16,.975, f"Cohort 1 | {LABELS[k]} | attacked B3, detected B{row.detected_building}", fontsize=9.5, weight="bold", va="top")
        fig.text(.16,.922, attacks[k], fontsize=9.4, va="top")
        ax.plot(h,d.clean,color=COLORS["truth"],lw=1.8,label="Clean")
        ax.plot(h,d.attacked,color=COLORS["attack"],lw=1.1,ls="--",label="Attacked")
        ax.plot(h,d.defended,color=COLORS["repair"],lw=1.15,ls=":",marker="o",ms=2.4,markevery=6,label="Defended")
        fig.legend(*ax.get_legend_handles_labels(),loc="upper center",bbox_to_anchor=(.58,.874),ncol=3,fontsize=9,frameon=False)
        ax.set_ylabel(["Price (currency/kWh)","Load (kWh/step)","PV (kWh/step)","Hour (1-24)"][k])
        if k == 3:
            ax.set_ylim(.5,24.5)
            ax.set_yticks([1,6,12,18,24])
        ax.grid(alpha=.16); ax.tick_params(labelbottom=False)
        err.plot(h,d.absolute_error_before,color=COLORS["attack"],ls="--",lw=1.1,label="Before")
        err.plot(h,d.absolute_error_after,color=COLORS["repair"],lw=1.3,label="After")
        m = d.retained_perturbed.to_numpy(dtype=bool)
        if m.any():
            err.scatter(h[m],d.absolute_error_after.to_numpy()[m],s=22,marker="x",color="#202020",linewidths=.8,zorder=5)
        err.set_ylabel("Abs. error");err.set_xlabel("Hours since attack start (first 72 of 168)",fontsize=9)
        err.set_xlim(0,71);err.set_xticks([0,24,48,71]);err.grid(alpha=.16)
        err.legend(loc="upper right",fontsize=8,frameon=False,ncol=2)
        defense = "Peer clock via conditional tool" if k == 3 and row.operation == "CONDITIONAL_MODEL" else defenses[row.operation]
        fig.text(.16,.026, f"{defense} | {row.edited_samples}/168 edits | graph {row.recorded_live_graph_seconds:.2f} s",fontsize=8.7)
        for ext in ["png","pdf"]:
            fig.savefig(out / f"{row.attack}.{ext}",dpi=240,bbox_inches="tight",metadata={"Creator":"CommunityGuard-AI measured case reporting"})
        plt.close(fig)
    (out / "figure_provenance.json").write_text(json.dumps(provenance,indent=2)+"\n")
    return out


def write_case_captions(code, run, paper):
    """Write concrete captions using the same verified case data as the plots."""
    cases, data, provenance, p = case_details(code, run)
    captions=[]
    labels=["price","load","pv","clock"]
    descriptions = [
        r"Building~3 price attack and conditional-model repair. The ridge predictor uses peer observations and weather/calendar context. Table~\ref{tab:case-comparison} reports the matched recovery, tokens, and latency.",
        r"Building~3 load attack. The energy-balance candidate $c=\max(0,n+\widetilde p)$ uses protected net import and received PV. Black crosses mark perturbed samples that the guard leaves unchanged.",
        r"Building~3 PV attack. The candidate $c=\max(0,\widetilde\ell-n)$ uses received load and protected net import; transferred-PV evidence controls acceptance. Zero-PV samples are unaffected by the multiplicative attack.",
        r"Building~3 clock attack. The selected conditional tool returns the peer-hour median for this channel. The error panel uses circular distance on the 24-hour clock."
    ]
    for k,r in cases.iterrows():
        captions += [r"\begin{figure}[t]",r"\centering\includegraphics[width=\columnwidth]{figures/"+r.attack+".pdf}","\\caption{"+descriptions[k]+"}",r"\label{fig:"+labels[k]+"}",r"\end{figure}"]
    (Path(paper)/"case_figures.tex").write_text("\n".join(captions)+"\n")
    lo,hi=cases.other_graph_seconds.min(),cases.other_graph_seconds.max()
    timing=("For Figures~\\ref{fig:price}--\\ref{fig:clock}, latency is measured after the complete 168-hour input is available. "
            "Forensic and planning times are client-measured Llama calls. Total graph time spans graph construction, numerical evidence, both calls, verification, and trace/output recording; it excludes trajectory collection, training, and truth-based scoring. "
            f"The remaining non-call work totals {1000*lo:.0f}--{1000*hi:.0f}~ms per example; its individual stages were not separately timed. "
            "These CPU measurements use six inference threads. The hourly time axis describes simulated attack exposure; the seconds reported in Table~\\ref{tab:case-comparison} describe offline processing, not delay from attack onset. Replay preserves the recorded live timings and reports its own execution duration separately.\n")
    (Path(paper)/"case_timing.tex").write_text(timing)
    return cases
