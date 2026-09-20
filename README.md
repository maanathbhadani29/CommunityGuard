# CommunityGuard-AI

**An agentic framework for selective telemetry repair in energy communities, with reproducible experiments and a live Llama comparison tutorial.**

CommunityGuard-AI connects measurement evidence, incident diagnosis, repair planning, and numerical verification. It identifies an inconsistent building/channel, proposes a repair tool, checks individual changes, and preserves the received values when no proposal passes verification. LangGraph manages the workflow, bounded retries, and execution records. Rule-based policies and Llama can provide diagnosis and planning within the same framework.

This repository accompanies *CommunityGuard-AI: Agentic AI for Cyber-Physical Defense of Energy Communities* by **Maanat Bhadani, Alaa Selim, and Junbo Zhao**.

[Paper PDF](manuscript/CommunityGuard_AI_Updated.pdf) · [Complete LaTeX source ZIP](manuscript/CommunityGuard_AI_LaTeX.zip)

## Choose a tutorial

| Notebook | Purpose | Llama service needed? |
|---|---|---|
| [Reproduction tutorial](CommunityGuard_AI_Updated_Tutorial.ipynb) | Explain the method and regenerate the paper experiment using saved numerical models and archived Llama responses | No |
| [Live comparison tutorial](CommunityGuard_AI_Live_Comparison.ipynb) | Compare fresh Llama responses with archived-response replay and rules on identical attacked inputs | Yes, local Ollama |

The notebooks run independently and share the included code and input files. You do not need to run the reproduction notebook before the live comparison. Keep both notebooks beside the `code/` and `manuscript/` folders.

## Quick start

1. Clone the repository or download and extract its files into a writable folder.
2. Open a notebook in Jupyter Notebook or JupyterLab using a **Python 3.13** kernel.
3. Clear existing outputs, then select **Restart Kernel and Run All**.

The setup cell installs missing pinned scientific packages in the selected kernel environment and prints progress. First setup needs internet. The default reproduction run subsequently uses local files and requires no API key, GPU, or model service.

If you need a separate environment, run the following in an Anaconda/Miniconda terminal:

```bash
conda create -n communityguard python=3.13 notebook ipykernel -y
conda activate communityguard
python -m ipykernel install --user --name communityguard --display-name "Python 3.13 (CommunityGuard)"
jupyter notebook
```

Select **Python 3.13 (CommunityGuard)** in the notebook. See the official [Conda environment guide](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) and [IPython kernel instructions](https://ipython.readthedocs.io/en/stable/install/kernel_install.html) for environment setup. Scientific package versions and installation logic are in [requirements-python313.txt](code/requirements-python313.txt) and [tutorial_setup.py](code/tutorial_setup.py); the setup helper also installs the specified CPU PyTorch version when needed.

## Reproduce the paper experiment

Open `CommunityGuard_AI_Updated_Tutorial.ipynb` with its default settings:

```python
FIT_MODE = "saved"
INFERENCE = "replay"
```

This mode restores trained numerical models and calibration parameters. It generates the attacks again, executes the defense graph, computes repairs and evaluation metrics, and creates the figures, tables, and manuscript values.

For the language component, replay uses an archived model response only when the newly generated prompt matches its original request exactly. A mismatch stops execution. Repair curves and target recovery scores are not substituted to force agreement. **Llama token counts and inference latency in replay remain historical measurements; the default does not retrain models or perform new Llama inference.**

The tutorial covers a worked load attack, the mathematical guard, numerical baselines, authentic Llama exchanges, a failed retry, and the method's failure cases. Figures 3–6 appear in individual display cells.

## Run fresh Llama inference

Install and open [Ollama](https://docs.ollama.com/windows). In a terminal, download the model:

```bash
ollama pull llama3.2:3b
```

If a background service is not already running, start `ollama serve` in a separate terminal. See the [Ollama CLI reference](https://docs.ollama.com/cli).

Open `CommunityGuard_AI_Live_Comparison.ipynb`. Its initial service check confirms that Ollama and the selected model are available. The live arm makes fresh requests and never substitutes replay responses or rule decisions when a call fails.

| Setting | Default | Alternative |
|---|---|---|
| `CASE_SCOPE` | `"figures_only"`: four attacks and one clean control | `"paper_subset"`: 48 attacks and four clean controls |
| `LIVE_FIT_MODE` | `"saved"`: keep numerical fits fixed to compare Llama behavior | `"refit"`: train new numerical models before fresh inference |
| `LLAMA_MODEL` | `"llama3.2:3b"` | Another installed model constitutes a different experiment |
| `REQUEST_TIMEOUT_SECONDS` | `180` | Adjust for the local inference runtime |

For the complete language-model subset, use:

```python
CASE_SCOPE = "paper_subset"
LIVE_FIT_MODE = "saved"
```

For fresh numerical training as well, set `LIVE_FIT_MODE = "refit"`. Training uses the included training/calibration CSV data; it does not collect a new CityLearn simulation. The reference arm retains the original fits and archived decisions, so this comparison can reflect both training and Llama changes.

The notebook compares selected targets, repair tools, repaired arrays, recovery errors, collateral errors, tokens, and timing. It plots fresh repairs beside the four paper cases and displays the actual prompts, responses, and graph traces. **A valid comparison may show better, worse, or identical live results.** It does not require matching the paper.

Progress messages identify each call, and a heartbeat reports elapsed time. Transport failures stop the batch and preserve completed cases. Rerun the live cell after fixing the service, or set `RESUME_FOLDER` to the same comparison folder after restarting the kernel. Resume supports saved-fit mode with unchanged settings and model/runtime versions. Use a new folder for a completely new run or for retraining.

## Results and validation

The reference numerical experiment evaluated **1,920 attacked records and 32 clean controls** across two cohorts. CommunityGuard-AI with rules improved **1,613** attacks, left **296** unchanged, and worsened **11**, reducing mean normalized error by **79.2%**. All 32 clean controls were left unchanged. In the original paired 48-attack study, rules improved 41 records and each Llama configuration improved 40.

Validation completed so far:

- **Reproduction:** all 27 code cells and 10 graph behavior tests passed on Linux/Python 3.13.15. An author-run Windows/Python 3.13.5 notebook also completed and matched the displayed scientific results.
- **Fresh Llama, illustrated cases:** an author-run Windows/Python 3.13.5 experiment with Ollama 0.34.0 and `llama3.2:3b` completed 16 fresh calls with no invalid responses. Both Llama configurations improved all four attacks and preserved the clean control. The notebook reported exact repaired-array equality with replay in all 10 case-policy comparisons.
- **Validation scope:** that fresh run used saved numerical fits and covered the four illustrated attacks plus one clean control. It did not establish live agreement across the full 48-attack subset or after numerical retraining. Offline tests separately exercised comparison logic, transport failures, resume behavior, and numerical retraining for one cohort.

Fresh response wording, token counts, and inference times can differ even when the selected tools and repaired arrays match. Hardware, model/runtime versions, loading, caching, and execution order affect timing. Archived and fresh timings should not be interpreted as a controlled algorithmic speed comparison. Very small nonzero errors may appear as `0.0` in rounded displays.

## Generated files

| Location | Contents |
|---|---|
| `code/communityguard/` | Readable implementation of models, attacks, agents, graph execution, evaluation, and plotting |
| `code/teaching_data/` | Cohort CSV inputs, column dictionaries, and data metadata |
| `code/protocols/` | Experiment settings, data splits, attack parameters, and trust assumptions |
| `code/paper_run/` | Saved numerical parameters, model weights, archived Llama exchanges, and historical timing records |
| `manuscript/` | Reference paper, manuscript template, and complete LaTeX ZIP |
| `results/run_*/` | Newly computed reproduction outputs and matching manuscript files |
| `results/live_compare_*/` | Live comparison metrics, differences, model details, raw exchanges, graph states, and plots |

The reproduction notebook exports a complete LaTeX ZIP for its run. If `pdflatex` is available, it also compiles a PDF; otherwise the figures, tables, and LaTeX sources are still generated. The included eight-page PDF is available immediately. The live notebook exports its comparison results separately and does not overwrite the manuscript.

## Data and assumptions

The included observations were collected from CityLearn 2.0.0 using the Challenge 2022 phase-1 and phase-2 cohorts with zero simulator actions. Each cohort contains 8,064 hourly observations for five buildings, split chronologically into 4,032 training, 1,344 calibration, and 2,688 test hours. The notebooks read the included observations directly. See the [CityLearn project](https://www.citylearn.net/) for the underlying simulation framework and its references.

The ordinary benchmark assumes one compromised price, load, PV, or clock channel in one building, honest neighboring buildings, and a protected net-electricity measurement. The load–PV balance applies to the stated zero-action simulation. The guard's error-reduction result is conditional on reference accuracy; calibration does not guarantee that condition for every future sample. These experiments evaluate offline telemetry repair and do not establish protection of physical equipment or recovery under unrestricted coordinated attacks.

## Troubleshooting

- **Project files not found:** open the notebook inside the extracted project folder, or set its `PROJECT_FOLDER` variable.
- **Package-version conflict:** restart the kernel and execute setup before importing numerical libraries.
- **Old paths or duplicate outputs:** clear all outputs, restart, and run the cells in order.
- **Ollama unavailable:** open the service, confirm the model is installed, and rerun the service check.
- **Replay mismatch:** a changed dataset, protocol, fit, or prompt may no longer correspond to the archived request. Use a separate fresh experiment for such changes.
- **No newly compiled PDF:** install a TeX distribution or compile the generated LaTeX ZIP separately. TeX is optional for running the experiments.

For research use, cite the accompanying CommunityGuard-AI manuscript and the relevant CityLearn references. Publication details should follow the version of the manuscript used.
