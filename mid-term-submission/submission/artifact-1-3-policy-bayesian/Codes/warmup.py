from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple
import numpy as np
import joblib
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel

@dataclass
class ParameterSetting:
    tx_power_dBm: float
    channel_id: int
    channel_width_MHz: int
    obss_pd: float
    target_rssi_dBm: float

def param_to_vec(p: ParameterSetting,
                 bounds: Dict[str, Tuple[float, float]],
                 enums: Dict[str, List]) -> np.ndarray:
    def norm_disc(val, allowed):
        idx = allowed.index(val)
        return idx / (len(allowed) - 1) if len(allowed) > 1 else 0.0
    return np.array([
        (p.tx_power_dBm - bounds["tx_power_dBm"][0])/(bounds["tx_power_dBm"][1]-bounds["tx_power_dBm"][0]),
        norm_disc(p.channel_id, enums["channel_id"]),
        norm_disc(p.channel_width_MHz, enums["channel_width_MHz"]),
        float(p.obss_pd),
        (p.target_rssi_dBm - bounds["target_rssi_dBm"][0])/(bounds["target_rssi_dBm"][1]-bounds["target_rssi_dBm"][0]),
    ], dtype=float)

def make_gp(ard_dims: int) -> GaussianProcessRegressor:
    kernel = 1.0 * Matern(length_scale=np.ones(ard_dims), length_scale_bounds=(1e-2, 1e3), nu=2.5) \
             + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e1))
    return GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=3, random_state=42)

def warmstart_and_save(context_id: str,
                       seeds: List[ParameterSetting],
                       rewards: List[float],
                       constraint_margins: Dict[str, List[float]],
                       bounds: Dict[str, Tuple[float, float]],
                       enums: Dict[str, List],
                       model_dir: str = "./models") -> str:
    X = np.vstack([param_to_vec(p, bounds, enums) for p in seeds])
    y = np.array(rewards, dtype=float)
    gp_reward = make_gp(X.shape[1]); gp_reward.fit(X, y)

    gp_constraints = {}
    for name, vals in constraint_margins.items():
        m = np.array(vals, dtype=float)
        gp = make_gp(X.shape[1]); gp.fit(X, m)
        gp_constraints[name] = gp

    state = {
        "context_id": context_id,
        "bounds": bounds,
        "enums": enums,
        "gp_reward": gp_reward,
        "gp_constraints": gp_constraints,
        "X": X, "y": y,
        "margins": {k: np.array(v, dtype=float) for k, v in constraint_margins.items()},
        "incumbent_y": float(np.max(y)),
        "incumbent_param": seeds[int(np.argmax(y))].__dict__,
    }
    path = f"{model_dir}/bo_state_{context_id}.pkl"
    joblib.dump(state, path)
    return path

