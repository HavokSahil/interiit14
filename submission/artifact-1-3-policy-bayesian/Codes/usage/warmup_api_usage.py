from warmup import warmstart_and_save, ParameterSetting

bounds = {"tx_power_dBm": (10.0, 20.0), "target_rssi_dBm": (-80.0, -65.0)}
enums = {"channel_id": [36,40,44,48], "channel_width_MHz": [20,40,80,160]}

seeds = [
    ParameterSetting(15.0, 36, 40, 0.4, -75.0),
    ParameterSetting(16.0, 44, 40, 0.5, -73.0),
]

rewards = [10.2, 11.1]
margins = {
    "g_coverage": [-0.8, -0.4],
    "g_util":     [-0.10, -0.08],
}

path = warmstart_and_save(
    context_id="test run 1",
    seeds=seeds,
    rewards=rewards,
    constraint_margins=margins,
    bounds=bounds,
    enums=enums,
    model_dir="./models"
)
print("Saved warmstart state at:", path)

