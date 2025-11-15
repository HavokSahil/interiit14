import os, json, csv, time, statistics
from pathlib import Path
from typing import List
from bo import SuggestInterface, InputSnapshot, ClientFeature, APFeature
from warmup import warmstart_and_save, ParameterSetting

MODEL_DIR = "./models"
STATE_FILE = Path(MODEL_DIR) / "bo_state_siteA.pkl"

def ensure_models():
    Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
    if not STATE_FILE.exists():
        # minimal warm start if user didn't run it yet
        seeds = [
            ParameterSetting(15.0, 36, 40, 0.35, -75.0),
            ParameterSetting(16.0, 44, 40, 0.45, -73.0),
            ParameterSetting(14.0, 40, 20, 0.25, -76.0)
        ]
        rewards = [-25.8, -24.9, -27.3]
        margins = {"g_coverage": [-0.8,-0.5,-0.3], "g_util": [-0.10,-0.07,-0.12]}
        bounds = {"tx_power_dBm": (10.0, 20.0), "target_rssi_dBm": (-80.0, -65.0)}
        enums = {"channel_id": [36,40,44,48], "channel_width_MHz": [20,40,80,160]}
        warmstart_and_save("siteA", seeds, rewards, margins, bounds, enums, model_dir=MODEL_DIR)

def build_snapshot(ctx_id: str, raw: dict) -> InputSnapshot:
    clients = [ClientFeature(**c) for c in raw["per_client_features"]]
    aps = [APFeature(**a) for a in raw["per_ap_features"]]
    return InputSnapshot(
        context_id=ctx_id,
        time_window=raw["time_window"],
        per_client_features=clients,
        per_ap_features=aps,
        environment_features=raw.get("environment_features"),
        policy_hyperparams=raw.get("policy_hyperparams"),
        constraint_thresholds=raw.get("constraint_thresholds"),
    )

def median_edge_dl(clients: List[ClientFeature]) -> float:
    edge = [c.bitrate_mbps for c in clients if -70.0 <= c.rssi_dBm <= -65.0]
    return statistics.median(edge) if edge else float("nan")

def estimated_gain_from_config(base_ap: APFeature, cfg: dict) -> float:
    # Heuristic mapping to mimic post-change throughput gains
    gain = 0.0
    if cfg["channel_width_MHz"] > base_ap.channel_width_MHz: gain += 0.08
    if cfg["tx_power_dBm"] > base_ap.tx_power_dBm + 0.8:     gain += 0.04
    if cfg["target_rssi_dBm"] >= -72.0:                      gain += 0.03
    if cfg["obss_pd"] > 0.60:                                gain -= 0.03  # aggressive reuse penalty on edges
    return max(0.00, min(0.25, gain))

def main():
    ensure_models()

    with open("input_snapshots_50.json","r") as f:
        data = json.load(f)
    context_id = data["context_id"]
    snaps = data["snapshots"]

    api = SuggestInterface(model_dir=MODEL_DIR, policy_tag="demo")

    # Logs
    jsonl = open("suggestions_50.jsonl","w")
    csvf = open("suggestions_50.csv","w", newline="")
    writer = csv.DictWriter(csvf, fieldnames=[
        "idx","start_iso","end_iso",
        "baseline_edge_mbps","estimated_post_edge_mbps","estimated_lift_pct",
        "tx_power_dBm","channel_id","channel_width_MHz","obss_pd","target_rssi_dBm",
        "safe_set_pass","feasibility_prob","score","expected_improvement"
    ])
    writer.writeheader()

    lifts = []
    for idx, snap_raw in enumerate(snaps, start=1):
        snapshot = build_snapshot(context_id, snap_raw)
        base_ap = snapshot.per_ap_features[0]
        baseline = median_edge_dl(snapshot.per_client_features)

        out = api.suggest(snapshot)
        cfg = out.parameter_setting
        gain = estimated_gain_from_config(base_ap, cfg)
        post = baseline*(1.0 + gain) if baseline == baseline else float("nan")  # NaN-safe

        # store for aggregate lift (ignore NaNs)
        if baseline == baseline and post == post and baseline > 0:
            lifts.append((post - baseline)/baseline)

        # Combined record
        record = {
            "input_snapshot": snap_raw,
            "output_configuration": cfg,
            "confidence": out.confidence,
            "validity": out.validity,
            "metadata": out.metadata,
            "baseline_edge_mbps": baseline,
            "estimated_post_edge_mbps": post,
            "estimated_lift_pct": None if baseline != baseline or post != post or baseline == 0 else 100.0*(post-baseline)/baseline
        }
        jsonl.write(json.dumps(record) + "\n")

        writer.writerow({
            "idx": idx,
            "start_iso": snap_raw["time_window"]["start_iso"],
            "end_iso": snap_raw["time_window"]["end_iso"],
            "baseline_edge_mbps": baseline,
            "estimated_post_edge_mbps": post,
            "estimated_lift_pct": None if baseline != baseline or post != post or baseline == 0 else 100.0*(post-baseline)/baseline,
            "tx_power_dBm": cfg["tx_power_dBm"],
            "channel_id": cfg["channel_id"],
            "channel_width_MHz": cfg["channel_width_MHz"],
            "obss_pd": cfg["obss_pd"],
            "target_rssi_dBm": cfg["target_rssi_dBm"],
            "safe_set_pass": out.validity.get("safe_set_pass"),
            "feasibility_prob": out.confidence.get("feasibility_prob"),
            "score": out.confidence.get("score"),
            "expected_improvement": out.confidence.get("expected_improvement")
        })

        # Feed a benign “measured” observation so the model updates between calls
        dummy_reward = -25.0 + 0.05*idx
        dummy_margins = {"g_coverage": -0.5, "g_util": -0.08}
        api.record_observation(context_id, cfg, dummy_reward, dummy_margins)

    jsonl.close(); csvf.close()

    # Aggregate QoE lift (median across snapshots with edge clients present)
    if lifts:
        med = statistics.median(lifts)
        print(f"Median edge-client DL throughput lift (estimated): {med*100:.1f}% over 50 suggestions.")
    else:
        print("No edge clients in snapshots; cannot compute lift.")

    print("Wrote suggestions_50.jsonl and suggestions_50.csv")

if __name__ == "__main__":
    main()

