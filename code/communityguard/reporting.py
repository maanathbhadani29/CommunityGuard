"""Readable scientific plots and exports from actual saved records."""
from __future__ import annotations
import csv
import json
from pathlib import Path
import numpy as np
from .core import CHANNELS

def write_csv(path, rows):
    if not rows:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_suffix(p.suffix + ".tmp")
    with temp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    temp.replace(p)

def tex_escape(value):
    mapping = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
               "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
               "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(mapping.get(c, c) for c in str(value))

def write_agent_appendix(root):
    root = Path(root)
    lines = [r"\section{Raw Agent Outputs from a New Local Execution}",
             "These outputs belong to the numerical experiment saved in this run folder."]
    found = False
    for path in sorted((root / "incidents").glob("*_llama.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        found = True
        lines.append(r"\subsection{" + tex_escape(record["metrics"]["case_id"]) + "}")
        for stage in ("forensics", "defense"):
            output = record[stage]
            if output.get("source") != "ollama_live":
                continue
            lines.append(r"\textbf{" + stage.capitalize() + " raw response.}")
            raw = output["response"]["message"]["content"]
            # Escaping every control character prevents model text becoming executable TeX.
            lines.append(r"\begin{quote}\footnotesize\raggedright")
            lines.extend(tex_escape(line) + r"\par" for line in raw.splitlines())
            lines.append(r"\end{quote}")
            lines.append("Parser fallback used: " + str(output.get("fallback_used", False)) + ".")
        m = record["metrics"]
        lines.append("Numerical check: " + ("accepted" if m["accepted"] else "failed") +
                     "; pre/post MSE: " + f"{m['mse_pre']:.6g}/{m['mse_post']:.6g}. " +
                     "This is a reconstruction check, not a physical-safety conclusion.")
    if not found:
        lines.append("No live Llama responses are present in this run. Deterministic-rule outputs are not LLM outputs.")
    (root / "appendix_agent_outputs.tex").write_text("\n\n".join(lines) + "\n", encoding="utf-8")

def plot_case(root, case, mode, loc, v, names, buildings):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    b, width = loc["building_index"], len(names)
    col = b*width + names.index(CHANNELS[case["attack"]])
    h = np.arange(len(case["x"]))
    fig, axes = plt.subplots(2, 1, figsize=(3.5, 2.9), sharex=True, constrained_layout=True)
    for values, label, color, style in ((case["clean"][:, col], "Clean reference", "#2b8c50", "--"),
             (case["x"][:, col], "Observed input", "#c74141", "-"),
             (v["defended"][:, col], "AE replacement", "#2468a0", "-.")):
        axes[0].plot(h, values, label=label, color=color, linestyle=style, linewidth=1.1)
    axes[0].set_ylabel("Standardized telemetry")
    axes[0].set_title(f"{case['attack'].replace('_',' ').title()} | {mode} | "
                      f"injected B{case['true_building']+1}, localized B{b+1}", fontsize=7)
    before = ((case["x"]-v["prediction_before"])**2).reshape(len(h), buildings, width).mean(2)[:, b]
    after = ((v["defended"]-v["prediction_after"])**2).reshape(len(h), buildings, width).mean(2)[:, b]
    axes[1].plot(h, before, label="Before", color="#8356b0", linewidth=1.1)
    axes[1].plot(h, after, label="After", color="#138e9d", linewidth=1.1)
    axes[1].axhline(float(before.mean()+2*before.std()), label="Attack-derived reference",
                   color="#bd7c12", linestyle=":", linewidth=1)
    axes[1].set_ylabel("Building MSE")
    axes[1].set_xlabel("Hour")
    for ax in axes:
        ax.grid(alpha=0.15)
        ax.legend(fontsize=5.7, loc="upper right", framealpha=0.9)
    p = Path(root) / "figures"
    p.mkdir(exist_ok=True)
    stem = f"{case['case_id']}_{mode}"
    write_csv(p / (stem + "_trace.csv"), [
        {"hour": int(t), "clean_telemetry": float(case["clean"][t, col]),
         "observed_telemetry": float(case["x"][t, col]),
         "replaced_telemetry": float(v["defended"][t, col]),
         "building_mse_before": float(before[t]), "building_mse_after": float(after[t]),
         "attack_derived_reference": float(before.mean()+2*before.std())} for t in h])
    for ext in ("pdf", "png"):
        metadata = {"CreationDate": None, "ModDate": None} if ext == "pdf" else None
        fig.savefig(p / f"{stem}.{ext}", dpi=220, metadata=metadata)
    plt.close(fig)
