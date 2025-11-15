import json, random
from datetime import datetime, timedelta

random.seed(42)

def mk_client(i, rssi, snr, rtt, jit, plr, retry, br):
    return {
        "client_id": f"cl{i}",
        "bssid": f"aa:bb:cc:dd:ee:{i:02d}",
        "rssi_dBm": rssi,
        "snr_dB": snr,
        "rtt_ms": rtt,
        "jitter_ms": jit,
        "plr": plr,
        "retries": retry,
        "bitrate_mbps": br
    }

def mk_ap(ch, cw, pwr, obss, trssi, util):
    return {
        "ap_id": "AP1",
        "channel_id": ch,
        "channel_width_MHz": cw,
        "tx_power_dBm": pwr,
        "obss_pd": obss,
        "target_rssi_dBm": trssi,
        "channel_utilization": util,
        "noise_floor_dBm": -95.0
    }

def jitter(val, lo, hi):
    return max(lo, min(hi, val + random.uniform(-0.05*(hi-lo), 0.05*(hi-lo))))

def main():
    context = "siteA"
    t0 = datetime.fromisoformat("2025-11-15T09:00:00")
    chans = [36, 40, 44, 48]
    cws = [20, 40, 80, 160]

    snapshots = []
    for k in range(50):
        ts = t0 + timedelta(minutes=5*k)
        tw = {
            "start_iso": (ts).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "end_iso":   (ts + timedelta(minutes=3)).strftime("%Y-%m-%dT%H:%M:%SZ")
        }

        # Create a mix of clients with some in the edge band (-70..-65)
        base_rssi = random.choice([-72,-71,-70,-69,-68,-67,-66,-65,-64])
        clients = []
        for i in range(1, 6):
            rssi = jitter(base_rssi + random.choice([-2,-1,0,1,2]), -80, -55)
            snr  = jitter(20 + (-(rssi+65))*0.6, 8, 35)
            rtt  = jitter(20 + (-(snr-20))*0.7, 8, 60)
            jit  = jitter(4.0 + (-(snr-20))*0.15, 0.5, 15)
            plr  = max(0.005, min(0.05, 0.02 + (20-snr)*0.002 + random.uniform(-0.003,0.003)))
            retry= max(0.03, min(0.15, 0.08 + (20-snr)*0.004 + random.uniform(-0.01,0.01)))
            br   = max(50, min(200, 100 + (snr-20)*3 + (random.random()-0.5)*15))
            clients.append(mk_client(i, rssi, snr, rtt, jit, plr, retry, br))

        ap = mk_ap(
            ch=random.choice(chans),
            cw=random.choice(cws),
            pwr=round(random.uniform(14.0, 17.0), 1),
            obss=round(random.uniform(0.25, 0.6), 2),
            trssi=round(random.uniform(-75.0, -70.0), 1),
            util=round(random.uniform(0.65, 0.82), 2)
        )

        snapshots.append({
            "time_window": tw,
            "per_client_features": clients,
            "per_ap_features": [ap],
            "environment_features": {"neighbor_ap_count": random.randint(4,9), "band":"5G","regulatory_domain":"US"},
            "policy_hyperparams": {
                "weights_alpha": { "rate": 0.4, "lat": 0.2, "jit": 0.1, "loss": 0.1, "retry": 0.1, "snr": 0.1 },
                "normalizers": { "R0": 100, "omega0": 50, "J0": 10, "Delta_snr": 5 },
                "snr_min_dB": 20
            },
            "constraint_thresholds": { "rssi_min_dBm": -72.0, "utilization_max": 0.85,
              "kpi_budgets": { "SNR<floor": 0.05, "PLR>thr": 0.02 } }
        })

    out = {"context_id": context, "snapshots": snapshots}
    with open("input_snapshots_50.json","w") as f:
        json.dump(out, f, indent=2)
    print("Wrote input_snapshots_50.json with 50 snapshots.")

if __name__ == "__main__":
    main()

