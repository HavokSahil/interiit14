from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import numpy as np
import joblib
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor

#defining "structs" as mentioned in schema
@dataclass
class ClientFeature:
    client_id: str
    bssid: str
    rssi_dBm: float
    snr_dB: float
    rtt_ms: float
    jitter_ms: float
    plr: float
    retries: float
    bitrate_mbps: float

@dataclass
class APFeature:
    ap_id: str
    channel_id: int
    channel_width_MHz: int
    tx_power_dBm: float
    obss_pd: float
    target_rssi_dBm: float
    channel_utilization: float
    noise_floor_dBm: float

@dataclass
class InputSnapshot:
    context_id: str # added this just in case
    time_window: Dict[str, str]
    per_client_features: List[ClientFeature]
    per_ap_features: List[APFeature]
    environment_features: Optional[Dict] = None
    policy_hyperparams: Optional[Dict] = None
    constraint_thresholds: Optional[Dict] = None

@dataclass
class OutputConfiguration:
    parameter_setting: Dict
    confidence: Dict
    validity: Dict
    metadata: Dict


class SuggestInterface:
    def __init__(self, model_dir: str = "./models", policy_tag: str = "prod"):
        self.model_dir = model_dir
        self.policy_tag = policy_tag

    def suggest(self, snapshot: InputSnapshot,
                n_candidates: int = 80,
                xi: float = 1e-3,
                beta: float = 2.0) -> OutputConfiguration:
        state = self._load_state(snapshot.context_id)

        bounds, enums = state["bounds"], state["enums"]
        gp_r: GaussianProcessRegressor = state["gp_reward"]
        gp_c: Dict[str, GaussianProcessRegressor] = state["gp_constraints"]
        y_star: float = state.get("incumbent_y", float(np.max(state["y"])))

        seed = self._seed_from_snapshot(snapshot)
        cands = self._sample_candidates(seed, bounds, enums, n_candidates)

        Xcand = np.vstack([self._param_to_vec(c, bounds, enums) for c in cands])

        mask = np.ones(len(cands), dtype=bool)
        mu_c, sd_c = {}, {}
        for name, gp in gp_c.items():
            mu, sd = gp.predict(Xcand, return_std=True)
            mu_c[name], sd_c[name] = mu, sd
            mask &= (mu + beta*sd) <= 0.0

        if not mask.any():
            return self._mk_output(seed, 0.0, 0.0, 0.0, False, ["no_safe_candidate"], snapshot)

        Xs = Xcand[mask]
        cs = [cands[i] for i, ok in enumerate(mask) if ok]

        mu_r, sd_r = gp_r.predict(Xs, return_std=True)
        sd_r = np.maximum(sd_r, 1e-9)
        z = (mu_r - y_star - xi) / sd_r
        ei = (mu_r - y_star - xi)*norm.cdf(z) + sd_r*norm.pdf(z)

        pfeas = np.ones(len(cs))
        for name in mu_c:
            mu_s = mu_c[name][mask]; sd_s = np.maximum(sd_c[name][mask], 1e-9)
            pfeas *= norm.cdf((0.0 - mu_s) / sd_s)

        eic = ei * pfeas
        pick = int(np.argmax(eic))
        best = cs[pick]

        return self._mk_output(best, float(eic[pick]), float(ei[pick]), float(pfeas[pick]), True, [], snapshot)

    def record_observation(self,
                           context_id: str,
                           applied_param: Dict,
                           reward: float,
                           margins: Dict[str, float]) -> None:
        state = self._load_state(context_id)
        bounds, enums = state["bounds"], state["enums"]

        x_new = self._param_to_vec(applied_param, bounds, enums).reshape(1, -1)
        state["X"] = np.vstack([state["X"], x_new])
        state["y"] = np.append(state["y"], float(reward))

        for name in state["gp_constraints"]:
            if name not in state["margins"]:
                state["margins"][name] = np.empty(0)
            state["margins"][name] = np.append(state["margins"][name], float(margins[name]))

        gp_r: GaussianProcessRegressor = state["gp_reward"]
        gp_r.fit(state["X"], state["y"])
        for name, gp in state["gp_constraints"].items():
            gp.fit(state["X"], state["margins"][name])

        if margins and all(v <= 0.0 for v in margins.values()) and float(reward) > state.get("incumbent_y", -np.inf):
            state["incumbent_y"] = float(reward)
            state["incumbent_param"] = dict(applied_param)

        self._save_state(context_id, state)


    def _load_state(self, context_id: str) -> Dict:
        path = f"{self.model_dir}/bo_state_{context_id}.pkl"
        state = joblib.load(path)
        return state

    def _save_state(self, context_id: str, state: Dict) -> None:
        path = f"{self.model_dir}/bo_state_{context_id}.pkl"
        joblib.dump(state, path)

    def _seed_from_snapshot(self, snap: InputSnapshot) -> Dict:
        if snap.per_ap_features:
            ap = snap.per_ap_features[0]
            return {
                "tx_power_dBm": ap.tx_power_dBm,
                "channel_id": ap.channel_id,
                "channel_width_MHz": ap.channel_width_MHz,
                "obss_pd": ap.obss_pd,
                "target_rssi_dBm": ap.target_rssi_dBm,
            }
            # TODO !!
            # modify this later to a known safe state, this is a fallback 
        return {"tx_power_dBm": 0.5, "channel_id": 36, "channel_width_MHz": 40, "obss_pd": 0.4, "target_rssi_dBm": -73.0}

    def _param_to_vec(self, p: Dict, bounds: Dict, enums: Dict) -> np.ndarray:
        def norm_disc(val, allowed):
            idx = allowed.index(val)
            return idx / (len(allowed) - 1) if len(allowed) > 1 else 0.0
        return np.array([
            (p["tx_power_dBm"] - bounds["tx_power_dBm"][0])/(bounds["tx_power_dBm"][1]-bounds["tx_power_dBm"][0]),
            norm_disc(p["channel_id"], enums["channel_id"]),
            norm_disc(p["channel_width_MHz"], enums["channel_width_MHz"]),
            float(np.clip(p["obss_pd"], 0.0, 1.0)),
            (p["target_rssi_dBm"] - bounds["target_rssi_dBm"][0])/(bounds["target_rssi_dBm"][1]-bounds["target_rssi_dBm"][0]),
        ], dtype=float)

    def _sample_candidates(self, seed: Dict, bounds: Dict, enums: Dict, n: int) -> List[Dict]:
        xs = []
        x0 = self._param_to_vec(seed, bounds, enums)
        for _ in range(n):
            jitter = np.clip(np.random.normal(0, 0.15, size=5), -0.4, 0.4)
            x = np.clip(x0 + jitter, 0.0, 1.0)
            xs.append(self._vec_to_param(x, seed, bounds, enums))
        return xs

    def _vec_to_param(self, x: np.ndarray, template: Dict, bounds: Dict, enums: Dict) -> Dict:
        tx = bounds["tx_power_dBm"][0] + x[0]*(bounds["tx_power_dBm"][1]-bounds["tx_power_dBm"][0])
        ch_idx = int(round(x[1]*(len(enums["channel_id"])-1)))
        cw_idx = int(round(x[2]*(len(enums["channel_width_MHz"])-1)))
        ob = float(np.clip(x[3], 0.0, 1.0))
        tr = bounds["target_rssi_dBm"][0] + x[4]*(bounds["target_rssi_dBm"][1]-bounds["target_rssi_dBm"][0])
        out = dict(template)
        out.update({
            "tx_power_dBm": float(tx),
            "channel_id": int(enums["channel_id"][ch_idx]),
            "channel_width_MHz": int(enums["channel_width_MHz"][cw_idx]),
            "obss_pd": ob,
            "target_rssi_dBm": float(tr),
        })
        return out

    def _mk_output(self, params: Dict, score: float, ei: float, pfeas: float,
                   valid: bool, violated: List[str], snapshot: InputSnapshot) -> OutputConfiguration:
        return OutputConfiguration(
            parameter_setting=params,
            confidence={"score": float(score), "expected_improvement": float(ei), "feasibility_prob": float(pfeas)},
            validity={"safe_set_pass": bool(valid), "violated_constraints": violated},
            metadata={"policy_tag": self.policy_tag, "generated_at_iso": snapshot.time_window.get("end_iso",""), "notes": "api-suggest"}
        )

