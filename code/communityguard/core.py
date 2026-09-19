"""Numerical operations. Scores are dimensionless, never physical power."""
from __future__ import annotations
from typing import Callable
import numpy as np

ATTACKS = ("market_hack", "meter_hack", "inverter_hack", "time_spoofing")
CHANNELS = {
    "market_hack": "electricity_pricing",
    "meter_hack": "non_shiftable_load",
    "inverter_hack": "solar_generation",
    "time_spoofing": "hour",
}
DIAGNOSES = dict(zip(ATTACKS, ("Market Hack", "Meter Hack", "Inverter Hack", "Time Spoofing")))

def mse(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or a.size == 0:
        raise ValueError("MSE inputs must have the same nonempty shape.")
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("MSE inputs contain a nonfinite value.")
    return float(np.mean((a - b) ** 2))

def inject_observation(obs, attack: str, target: int, names: list[str], rng) -> np.ndarray:
    """Copy one (building,feature) observation and corrupt one specified entry."""
    out = np.array(obs, dtype=np.float32, copy=True)
    if attack == "baseline":
        return out
    if attack not in CHANNELS or not 0 <= target < out.shape[0]:
        raise ValueError("Invalid attack class or building index.")
    k = names.index(CHANNELS[attack])
    v = float(out[target, k])
    if attack == "market_hack":
        out[target, k] = v * 0.60 + rng.normal(0, 0.02)
    elif attack == "meter_hack":
        out[target, k] = v * 0.70
    elif attack == "inverter_hack":
        out[target, k] = v * rng.uniform(0.1, 0.5)
    else:
        out[target, k] = (v - 4) % 24
    return out

def localize(x: np.ndarray, prediction: np.ndarray, buildings: int, names: list[str]) -> dict:
    if x.shape != prediction.shape or x.ndim != 2 or x.shape[1] != buildings * len(names):
        raise ValueError("The feature layout does not match the model observations.")
    residual = (x - prediction) ** 2
    building_mse = residual.reshape(len(x), buildings, len(names)).mean(axis=2)
    peak, building = np.unravel_index(np.argmax(building_mse), building_mse.shape)
    keys = ("Electricity_Pricing", "Non_Shiftable_Load", "Solar_Generation", "Clock_Sync")
    features = ("electricity_pricing", "non_shiftable_load", "solar_generation", "hour")
    sensors = {k: float(residual[peak, building * len(names) + names.index(f)])
               for k, f in zip(keys, features)}
    return {"peak_hour": int(peak), "building_index": int(building),
            "sensors": sensors, "building_mse": building_mse}

def recognized_protocol(command: str) -> bool:
    # Deliberately reproduces the notebook's substring semantics.
    return isinstance(command, str) and ("OVERRIDE" in command or "ISOLATION" in command)

def replace_and_verify(x: np.ndarray, predict: Callable, building: int,
                       features_per_building: int, command: str, clean=None) -> dict:
    if x.ndim != 2 or x.shape[1] % features_per_building:
        raise ValueError("Invalid feature layout.")
    if not 0 <= building < x.shape[1] // features_per_building:
        raise ValueError("Invalid localized building.")
    pred = np.asarray(predict(x))
    pre = mse(x, pred)
    y = x.copy()
    start = building * features_per_building
    if recognized_protocol(command):
        y[:, start:start + features_per_building] = pred[:, start:start + features_per_building]
    post_pred = np.asarray(predict(y))
    post = mse(y, post_pred)
    result = {"defended": y, "prediction_before": pred, "prediction_after": post_pred,
              "mse_pre": pre, "mse_post": post, "delta_mse": post - pre,
              "scaled_score_pre": 15.5 * pre, "scaled_score_post": 15.5 * post,
              "accepted": bool(post < pre), "recognized_protocol": recognized_protocol(command)}
    if clean is not None:
        result.update(clean_mse_pre=mse(x, clean), clean_mse_post=mse(y, clean))
    return result

def deterministic_diagnosis(sensors: dict) -> str:
    """Explicit comparator: map the largest of the same four residuals to a label."""
    mapping = {"Electricity_Pricing": "Market Hack", "Non_Shiftable_Load": "Meter Hack",
               "Solar_Generation": "Inverter Hack", "Clock_Sync": "Time Spoofing"}
    return mapping[max(sensors, key=sensors.get)]

def conditional_bound(residual_norm: float, replaced_residual_norm: float, lipschitz: float) -> float:
    """Appendix bound on post/pre residual norm, for a supplied valid Lipschitz bound."""
    if residual_norm <= 0 or not 0 <= replaced_residual_norm <= residual_norm or lipschitz < 0:
        raise ValueError("Invalid bound inputs.")
    q = replaced_residual_norm / residual_norm
    return float(np.sqrt(max(0.0, 1.0 - q*q)) + lipschitz*q)

