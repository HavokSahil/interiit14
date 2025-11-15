# demo_use_interface.py
from __future__ import annotations
from interface_api import SuggestInterface, InputSnapshot, ClientFeature, APFeature

def make_placeholder_snapshot(context_id: str) -> InputSnapshot:
    return InputSnapshot(
        context_id=context_id,
        time_window={"start_iso":"2025-11-13T00:10:00Z","end_iso":"2025-11-13T00:13:00Z"},
        per_client_features=[
            ClientFeature("cl1","aa",-67,24,22,4,0.01,0.05,120),
            ClientFeature("cl2","aa",-70,21,28,5,0.015,0.08,95),
        ],
        per_ap_features=[
            APFeature("AP1", channel_id=36, channel_width_MHz=40, tx_power_dBm=16.0,
                      obss_pd=0.4, target_rssi_dBm=-73.0, channel_utilization=0.72, noise_floor_dBm=-95.0)
        ],
        constraint_thresholds={"rssi_min_dBm": -72.0, "utilization_max": 0.85}
    )

def deploy_apply(config: dict) -> None:
    pass

def measure_reward_and_margins() -> tuple[float, dict]:
    pass

def main():
    test_run = "test run 1"
    model_dir = "./models" 
    api = SuggestInterface(model_dir=model_dir, policy_tag="prod")

    snapshot = make_dummy_snapshot(context)
    suggestion = api.suggest(snapshot)

    print("Suggested config:", suggestion.parameter_setting)
    print("Confidence:", suggestion.confidence)
    print("Validity:", suggestion.validity)

    deploy_apply(suggestion.parameter_setting)
    reward, margins = measure_reward_and_margins()
    print("Measured reward:", reward, "margins:", margins)

    api.record_observation(
        context_id=context,
        applied_param=suggestion.parameter_setting,
        reward=reward,
        margins=margins
    )
    print("Model recorded and data augmented")

if __name__ == "__main__":
    main()

