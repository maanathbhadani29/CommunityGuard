# CommunityGuard-AI — choose a tutorial

| Notebook | What it runs |
|---|---|
| `CommunityGuard_AI_Updated_Tutorial.ipynb` | Validated paper reproduction with saved numerical fits and archived Llama responses; visible reference outputs included |
| `CommunityGuard_AI_Live_Comparison.ipynb` | Fresh Llama inference compared with replay and rules on matched inputs; requires local Ollama |

Both notebooks run independently from this extracted folder. They share the
included code and inputs; you do not need to run the first before the second.

## Live comparison quick start

1. Install/open Ollama; in a terminal run `ollama pull llama3.2:3b`.
2. Open `CommunityGuard_AI_Live_Comparison.ipynb` using a Python 3.13 kernel.
3. Clear outputs, restart, and run the cells in order. The notebook checks the
   local service before starting the experiment.
4. The default `CASE_SCOPE="figures_only"` compares four attacks and one clean
   control. Use `"paper_subset"` for all 48 attacks and four clean controls.

The live arm always requests new model responses. `LIVE_FIT_MODE="saved"` keeps
the numerical fits fixed for a direct Llama comparison; `"refit"` trains them
again from the included CSV, so differences can reflect both training and
Llama changes. The code reports the actual differences and never requires a
specific improvement rate or matching paper values.

Progress prints during calls. If transport fails, the batch stops and preserves
completed cases; it never switches to replay. Rerun the live cell after fixing
Ollama, or set `RESUME_FOLDER` to the same run folder after a kernel restart.
Resume is supported for saved-fit mode with unchanged settings/model version.
Model loading and hardware affect runtime; the complete subset takes longer.

No measured live results are supplied in advance. The new workflow was tested
offline for comparison and failure behavior; genuine Ollama inference must be
run on your machine. See the notebook's official Ollama links for setup.

---

# CommunityGuard-AI tutorial

## Start here

1. Extract the whole ZIP into a writable folder.
2. Open `CommunityGuard_AI_Updated_Tutorial.ipynb` in Jupyter Notebook or JupyterLab.
3. Select a **Python 3.13** kernel.
4. Clear all cell outputs, then **Restart Kernel and Run All**. Save when it finishes.

Keep the notebook, `code/`, and `manuscript/` together. Share this complete
small ZIP. No files from the earlier large archive are needed.

The notebook already shows example tables, responses, and plots. The four paper
plots each have their own display cell under **Paper Figures 3–6**. Read the
included eight-page paper in `manuscript/CommunityGuard_AI_Updated.pdf`.

## What this teaches

Follow a Building 3 load attack through evidence construction, channel diagnosis,
tool selection, numerical verification, and selective repair. Then run the full
benchmark, compare the baselines, inspect genuine Llama exchanges and a rejected
retry, and examine the framework's assumptions and failure cases. The worked
example can be modified independently of the fixed paper experiment.

## Setup and run modes

First setup needs internet to install missing pinned Python packages. The setup
cell lists the versions and prints progress. It uses the selected kernel's
environment; a separate Python 3.13 environment is useful if you also run
unrelated projects. Once packages are installed, the default experiment uses
the included local inputs and needs no model service, API key, or GPU.

The tested defaults are `FIT_MODE="saved"` and `INFERENCE="replay"`:

- Saved fits restore previously learned coefficients, calibration parameters,
  and standard PyTorch weight tensors.
- Replay uses genuine archived Llama responses only when the newly generated
  prompt matches the archived request exactly.
- Attacks, numerical baselines, graph execution, repairs, scores, and plots are
  computed again. No target repair curve is substituted.
- Llama tokens and model latency remain measurements of the original live
  calls. Current execution time is reported separately.

Fresh training or live Llama inference is a new experiment. Live inference needs
Ollama and the named model; changing the fits or prompts may invalidate replay.
The default workflow is the configuration validated for this release.

## Results and paper

Every execution creates a separate `results/run_*/` folder containing metrics,
response arrays, graph traces, evidence, figures, and generated LaTeX. Successful
execution ends with **Within-run checks: PASS**, **Paper/results agreement: PASS**,
and **TUTORIAL COMPLETE**. The ten graph behavior tests also report **OK**.

The matching reference LaTeX ZIP is included in `manuscript/`.
The final cell exports the complete LaTeX ZIP for each new run. If `pdflatex` is available, it also
compiles a PDF. Without TeX, the plots, tables, and LaTeX still run; the included
reference PDF can be read immediately. The original input CSV/JSON files and
ordinary Python source are all inspectable.

If old paths or duplicate outputs appear, clear all outputs before restarting
and rerunning. If setup asks for a restart because another package version is
already loaded, restart and run from the first cell. Run in the extracted folder,
not from inside the ZIP. Setup can take longer than the experiment itself.

## Validation and interpretation

`VALIDATION.json` records the fresh release check. The author's earlier uploaded
Windows/Python 3.13.5 run also completed all 27 code cells and matched the displayed
scientific results; the fresh release check was performed on Linux/Python 3.13.15.
Small floating-point and wall-clock differences can occur across machines.

The reference experiment improved 1,613 of 1,920 attacked records, left 296
unchanged, harmed 11, and reduced mean normalized error by 79.2%. All 32 clean
controls were left unchanged. The paired 48-attack study improved 41 cases with
rules and 40 with each Llama configuration. These are empirical results under
the stated single-channel/protected-meter assumptions, not universal recovery
or physical-equipment protection guarantees.

Reproduction-notebook validation completed all 27 cells in 61.2 seconds
with dependencies already installed. Installation time is additional.
