"""CityLearn collection, PyTorch training, and the four-node LangGraph runner."""
from __future__ import annotations
from datetime import datetime, timezone
import importlib.metadata
import json
import platform
import random
import time
from pathlib import Path
from typing import Any, TypedDict
import numpy as np
from .agents import OllamaClient, write_json
from .core import (ATTACKS, CHANNELS, DIAGNOSES, inject_observation, mse, localize,
                   replace_and_verify, deterministic_diagnosis, conditional_bound)
from .prompts import forensic_prompt, defense_prompt

class IncidentState(TypedDict, total=False):
    case_id: str
    mode: str
    x: Any
    clean: Any
    true_building: int
    attack: str
    localization: dict
    forensics: dict
    defense: dict
    verification: dict

def dependency_versions():
    out = {}
    for name in ("numpy", "pandas", "scikit-learn", "torch", "torchvision", "citylearn",
                 "gym", "matplotlib", "langgraph"):
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out

def collect(env, hours, names, attack, building, rng):
    value = env.reset()
    obs = value[0] if isinstance(value, tuple) else value
    histories, clean = [], []
    for step in range(hours):
        a = np.asarray(obs, dtype=np.float32)
        if a.ndim != 2 or a.shape[1] != len(names) or not np.isfinite(a).all():
            raise ValueError("Expected finite, equal-width observation vectors for all buildings.")
        clean.append(a.copy().reshape(-1))
        histories.append(inject_observation(a, attack, building, names, rng).reshape(-1))
        spaces = env.action_space
        if isinstance(spaces, (list, tuple)):
            actions = [np.zeros(s.shape, dtype=np.float32) for s in spaces]
        else:
            actions = np.zeros(spaces.shape, dtype=np.float32)
        value = env.step(actions)
        obs = value[0]
        if len(value) == 5:
            done = bool(value[2]) or bool(value[3])
        else:
            done = bool(value[2])
        if done and step < hours - 1:
            raise ValueError(f"CityLearn episode ended after {step+1} observations; lower --hours.")
    return np.asarray(histories, dtype=np.float32), np.asarray(clean, dtype=np.float32)

def build_model(x, epochs, learning_rate, threads):
    import torch
    from torch import nn
    torch.set_num_threads(threads)
    model = nn.Sequential(nn.Linear(x.shape[1], 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(),
                          nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, x.shape[1]))
    data = torch.tensor(x, dtype=torch.float32)
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    losses = []
    for _ in range(epochs):
        opt.zero_grad()
        loss = torch.mean((model(data) - data)**2)
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
    model.eval()
    def predict(value):
        with torch.no_grad():
            return model(torch.as_tensor(value, dtype=torch.float32)).cpu().numpy()
    norm_product = float(np.prod([np.linalg.norm(layer.weight.detach().numpy(), 2)
                                for layer in model if isinstance(layer, nn.Linear)]))
    return model, predict, losses, norm_product

def graph_runner(predict, buildings, names, client, root, prompt_profile="current"):
    from langgraph.graph import StateGraph, START, END

    def detect(state):
        return {"localization": localize(state["x"], predict(state["x"]), buildings, names)}

    def forensics(state):
        loc = state["localization"]
        if state["mode"] == "llama":
            log = root / "agent_logs" / (state["case_id"] + "_forensics.json")
            rec = client.invoke(forensic_prompt(loc["building_index"], loc["sensors"]), "forensics", log)
        else:
            rec = {"source": "deterministic_rule", "latency_seconds": 0.0, "total_tokens": 0,
                   "parsed": {"diagnosis": deterministic_diagnosis(loc["sensors"]),
                              "reasoning": "Deterministic mapping from the largest supplied sensor residual."},
                   "fallback_used": False}
        return {"forensics": rec}

    def defense(state):
        if state["mode"] == "llama":
            loc = state["localization"]
            log = root / "agent_logs" / (state["case_id"] + "_defense.json")
            rec = client.invoke(defense_prompt(loc["building_index"], state["forensics"]["parsed"]["diagnosis"], prompt_profile),
                                "defense", log)
        else:
            rec = {"source": "deterministic_rule", "latency_seconds": 0.0, "total_tokens": 0,
                   "parsed": {"command": "DIGITAL_TWIN_OVERRIDE", "justification": "Automatic comparator response."},
                   "fallback_used": False}
        return {"defense": rec}

    def verify(state):
        command = state["defense"]["parsed"]["command"]
        return {"verification": replace_and_verify(state["x"], predict,
                state["localization"]["building_index"], len(names), command, clean=state["clean"])}

    graph = StateGraph(IncidentState)
    for name, func in (("detection", detect), ("forensic_agent", forensics),
                       ("defense_agent", defense), ("verification_node", verify)):
        graph.add_node(name, func)
    graph.add_edge(START, "detection")
    graph.add_edge("detection", "forensic_agent")
    graph.add_edge("forensic_agent", "defense_agent")
    graph.add_edge("defense_agent", "verification_node")
    graph.add_edge("verification_node", END)
    return graph.compile()

def run(args):
    import torch
    from sklearn.preprocessing import StandardScaler
    from citylearn.citylearn import CityLearnEnv
    from .reporting import write_csv, plot_case, write_agent_appendix

    started = time.perf_counter()
    root = Path(args.output).resolve()
    # Never mix a fresh experiment with earlier results.
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Output folder is not empty: {root}. Choose a new --output folder.")
    root.mkdir(parents=True, exist_ok=True)
    client = None
    ollama_meta = None
    if args.agent in ("llama", "both"):
        client = OllamaClient(args.ollama_url, args.model, args.seed, args.temperature,
                              args.timeout, args.json_mode, args.prompt_profile)
        ollama_meta = client.preflight()

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    env = CityLearnEnv(schema=args.schema, central_agent=False)
    try:
        name_groups = env.observation_names
        names = list(name_groups[0])
        if any(list(v) != names for v in name_groups):
            raise ValueError("The paper model requires the same observation layout in every building.")
        missing = set(CHANNELS.values()) - set(names)
        if missing:
            raise ValueError(f"Required CityLearn observations are missing: {sorted(missing)}")
        buildings = len(name_groups)
        manifest = {"implementation": "communityguard-companion-1.2.0", "source_commit":
                    ("4a428fbeced2e6cef898a037fd5739d464e43e0e" if args.prompt_profile == "current"
                     else "68d870aa1435eaad2d560a7b413c56979e293a96"), "result_origin": "new_local_execution",
                    "created_utc": datetime.now(timezone.utc).isoformat(), "arguments": vars(args),
                    "python": platform.python_version(), "platform": platform.platform(),
                    "processor": platform.processor(), "packages": dependency_versions(),
                    "ollama": ollama_meta, "buildings": buildings, "features_per_building": len(names),
                    "observation_names": names, "actions": "zero", "status": "running",
                    "limits": "Offline same-horizon telemetry case study; no physical-safety certificate."}
        write_json(root / "manifest.json", manifest)
        (root / "data").mkdir(exist_ok=True)
        print(f"[1/5] CityLearn: {buildings} buildings, {len(names)} features, {args.hours} hours.", flush=True)
        raw, _ = collect(env, args.hours, names, "baseline", 0, np.random.default_rng(args.seed))
        scaler = StandardScaler()
        training = scaler.fit_transform(raw).astype(np.float32)
        np.savez_compressed(root / "data" / "baseline_and_scaler.npz", raw=raw, standardized=training,
                            mean=scaler.mean_, scale=scaler.scale_, var=scaler.var_)
        print("[2/5] Training the D-64-32-64-D autoencoder.", flush=True)
        model, predict, losses, norm_product = build_model(training, args.epochs, args.learning_rate, args.threads)
        torch.save({"state_dict": model.state_dict(), "input_width": training.shape[1],
                    "seed": args.seed}, root / "autoencoder.pt")
        write_json(root / "training.json", {"loss": losses,
                     "estimated_product_of_spectral_norms": norm_product,
                     "note": "Numerical norm estimate, not an outward-rounded formal certificate."})
        print("[3/5] Screening all fixed-magnitude attack/building candidates.", flush=True)
        candidates, selected = [], []
        for ai, attack in enumerate(ATTACKS):
            best = None
            for building in range(buildings):
                attack_seed = args.seed + 1000*(ai + 1) + building
                raw_attack, raw_clean = collect(env, args.hours, names, attack, building,
                                               np.random.default_rng(attack_seed))
                x = scaler.transform(raw_attack).astype(np.float32)
                clean = scaler.transform(raw_clean).astype(np.float32)
                score = mse(x, predict(x))
                case = {"attack": attack, "true_building": building, "x": x, "clean": clean,
                        "case_id": f"{attack}_b{building+1}", "screening_mse": score}
                np.savez_compressed(root / "data" / (case["case_id"] + ".npz"),
                                    raw_attack=raw_attack, raw_clean=raw_clean, attacked=x, clean=clean)
                candidates.append({"case_id": case["case_id"], "attack": attack,
                                   "injected_building": building+1, "rng_seed": attack_seed,
                                   "screening_mse": score})
                if args.scenarios == "all":
                    selected.append(case)
                if best is None or score > best["screening_mse"]:
                    best = case
            if args.scenarios == "selected":
                selected.append(best)
        write_csv(root / "screening.csv", candidates)
        write_json(root / "selected_cases.json", [
            {k:v for k,v in c.items() if k not in ("x", "clean")} for c in selected])
        modes = ("llama", "rules") if args.agent == "both" else (args.agent,)
        app = graph_runner(predict, buildings, names, client, root, args.prompt_profile)
        metrics, paired = [], []
        print(f"[4/5] Running {len(selected)} cases with agent mode(s): {', '.join(modes)}.", flush=True)
        for case in selected:
            verified_modes = {}
            for mode in modes:
                print(f"  {case['case_id']} / {mode}", flush=True)
                state = app.invoke({k:v for k,v in case.items() if k != "screening_mse"} | {"mode": mode})
                v, loc = state["verification"], state["localization"]
                f, d = state["forensics"], state["defense"]
                toks = (f["total_tokens"], d["total_tokens"])
                tokens = sum(toks) if all(t is not None for t in toks) else None
                r = case["x"] - v["prediction_before"]
                start = loc["building_index"] * len(names)
                rnorm = float(np.linalg.norm(r))
                masked_norm = min(rnorm, float(np.linalg.norm(r[:, start:start+len(names)])))
                row = {"case_id": case["case_id"], "mode": mode, "attack": case["attack"],
                       "injected_building": case["true_building"]+1,
                       "localized_building": loc["building_index"]+1, "peak_hour": loc["peak_hour"],
                       "localization_correct": loc["building_index"] == case["true_building"],
                       "diagnosis": f["parsed"]["diagnosis"], "command": d["parsed"]["command"],
                       "diagnosis_reasoning": f["parsed"].get("reasoning", ""),
                       "strategy_text": d["parsed"].get("strategy", ""),
                       "defense_justification": d["parsed"].get("justification", ""),
                       "forensic_fallback": f.get("fallback_used", False),
                       "defense_fallback": d.get("fallback_used", False),
                       "forensics_latency_seconds": f["latency_seconds"],
                       "defense_latency_seconds": d["latency_seconds"],
                       "total_llm_latency_seconds": f["latency_seconds"] + d["latency_seconds"],
                       "total_llm_tokens": tokens,
                       **{k:v[k] for k in ("mse_pre", "mse_post", "delta_mse", "scaled_score_pre",
                                           "scaled_score_post", "accepted", "recognized_protocol",
                                           "clean_mse_pre", "clean_mse_post")},
                       "conditional_norm_bound_estimate": conditional_bound(rnorm, masked_norm, norm_product)
                           if rnorm > 0 else None}
                # All numerical results and raw agent outputs remain together in this run folder.
                metrics.append(row)
                write_csv(root / "metrics.csv", metrics)
                record = {"result_origin": "new_local_execution", "mode": mode,
                          "metrics": row, "localized_sensors": loc["sensors"],
                          "forensics": f, "defense": d}
                write_json(root / "incidents" / f"{case['case_id']}_{mode}.json", record)
                np.savez_compressed(root / "data" / f"{case['case_id']}_{mode}_defended.npz",
                                    defended=v["defended"], predicted_before=v["prediction_before"],
                                    predicted_after=v["prediction_after"])
                if not args.no_plots:
                    plot_case(root, case, mode, loc, v, names, buildings)
                verified_modes[mode] = v
            if len(verified_modes) == 2:
                a, b = verified_modes["llama"], verified_modes["rules"]
                paired.append({"case_id": case["case_id"],
                    "llama_recognized_protocol": a["recognized_protocol"],
                    "max_defended_difference": float(np.max(np.abs(a["defended"]-b["defended"]))),
                    "llama_mse_post": a["mse_post"], "rules_mse_post": b["mse_post"],
                    "accepted_equal": a["accepted"] == b["accepted"]})
        if paired:
            write_csv(root / "paired_comparison.csv", paired)
        print("[5/5] Exporting full agent records and the LaTeX appendix.", flush=True)
        write_agent_appendix(root)
        manifest.update(status="completed", screened_candidates=len(candidates), selected_cases=len(selected),
                        evaluated_modes=list(modes), elapsed_seconds=time.perf_counter()-started)
        write_json(root / "manifest.json", manifest)
        print(f"Complete: {root}", flush=True)
        return root
    except Exception as exc:
        if "manifest" in locals():
            manifest.update(status="failed", error=str(exc), elapsed_seconds=time.perf_counter()-started)
            write_json(root / "manifest.json", manifest)
        raise
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()
