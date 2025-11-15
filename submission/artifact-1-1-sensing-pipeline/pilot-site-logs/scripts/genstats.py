import json
import sys
from collections import defaultdict

def main():
    if len(sys.argv) < 2:
        print("Usage: python generate_summary.py <file.json>")
        return

    path = sys.argv[1]
    with open(path, "r") as f:
        data = json.load(f)

    total_scans = 0
    freq_counts = defaultdict(int)
    total_dwell_time = 0.0
    timestamps = []

    snr_vals = []
    noise_vals = []

    reward_sums = []
    reward_counts = 0

    # Only summary per label
    label_stats = {}

    def ensure(lbl):
        if lbl not in label_stats:
            label_stats[lbl] = {
                "count": 0,

                "duty_sum": 0.0,
                "duty_min": None,
                "duty_max": None,

                "conf_sum": 0.0,
                "conf_min": None,
                "conf_max": None,

                "bw_sum": 0.0,
                "bw_min": None,
                "bw_max": None,
            }
        return label_stats[lbl]

    # --------- main loop ---------
    for epoch in data:
        rewards = epoch.get("actual_rewards", [])
        reward_sums.extend(rewards)
        reward_counts += len(rewards)

        for scan in epoch.get("scans", []):
            total_scans += 1

            cf = scan["center_frequency_hz"]
            freq_counts[cf] += 1
            total_dwell_time += scan.get("dwell_time", 0)

            ts = scan.get("timestamp")
            if ts is not None:
                timestamps.append(ts)

            snr_vals.append(scan.get("snr_db"))
            noise_vals.append(scan.get("noise_floor_dbm"))

            preds = scan.get("nonwifi_classifier_predictions", [])
            for p in preds:
                lbl = p.get("predicted_label", "UNKNOWN")
                st = ensure(lbl)

                st["count"] += 1

                # duty cycle
                dc = p.get("duty_cycle")
                if isinstance(dc, (int, float)):
                    st["duty_sum"] += dc
                    st["duty_min"] = dc if st["duty_min"] is None else min(st["duty_min"], dc)
                    st["duty_max"] = dc if st["duty_max"] is None else max(st["duty_max"], dc)

                # confidence
                conf = p.get("confidence")
                if isinstance(conf, (int, float)):
                    st["conf_sum"] += conf
                    st["conf_min"] = conf if st["conf_min"] is None else min(st["conf_min"], conf)
                    st["conf_max"] = conf if st["conf_max"] is None else max(st["conf_max"], conf)

                # bandwidth
                bw = p.get("bandwidth_hz")
                if isinstance(bw, (int, float)):
                    st["bw_sum"] += bw
                    st["bw_min"] = bw if st["bw_min"] is None else min(st["bw_min"], bw)
                    st["bw_max"] = bw if st["bw_max"] is None else max(st["bw_max"], bw)

    # --------- finalize summary ---------
    timestamps = [t for t in timestamps if t is not None]
    timespan = (max(timestamps) - min(timestamps)) if timestamps else 0

    # compute averages
    final_label_stats = {}
    for lbl, st in label_stats.items():
        c = st["count"]
        final_label_stats[lbl] = {
            "count": c,

            "avg_duty_cycle": (st["duty_sum"] / c) if c else None,
            "duty_min": st["duty_min"],
            "duty_max": st["duty_max"],

            "avg_confidence": (st["conf_sum"] / c) if c else None,
            "confidence_min": st["conf_min"],
            "confidence_max": st["conf_max"],

            "avg_bandwidth": (st["bw_sum"] / c) if c else None,
            "bandwidth_min": st["bw_min"],
            "bandwidth_max": st["bw_max"],
        }

    summary = {
        "num_epochs": len(data),
        "total_scans": total_scans,
        "distinct_frequencies": len(freq_counts),
        "frequency_counts": dict(freq_counts),

        "total_dwell_time_seconds": total_dwell_time,
        "wall_clock_timespan_seconds": timespan,

        "reward_stats": {
            "count": reward_counts,
            "sum": sum(reward_sums),
            "avg": (sum(reward_sums) / reward_counts) if reward_counts else None,
            "min": min(reward_sums) if reward_sums else None,
            "max": max(reward_sums) if reward_sums else None,
        },

        "label_stats": final_label_stats,

        "snr_db": {
            "min": min(snr_vals) if snr_vals else None,
            "max": max(snr_vals) if snr_vals else None,
            "avg": (sum(snr_vals) / len(snr_vals)) if snr_vals else None,
        },

        "noise_floor_dbm": {
            "min": min(noise_vals) if noise_vals else None,
            "max": max(noise_vals) if noise_vals else None,
            "avg": (sum(noise_vals) / len(noise_vals)) if noise_vals else None,
        }
    }

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

