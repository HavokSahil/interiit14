import json
import sys
import os
import matplotlib.pyplot as plt


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)


def save_plot(name):
    plt.tight_layout()
    plt.savefig(f"plots/{name}.png")
    plt.close()


def plot_frequency_counts(freq_counts):
    freqs = list(freq_counts.keys())
    counts = list(freq_counts.values())

    plt.figure()
    plt.bar(freqs, counts)
    plt.title("Scan Count per Frequency")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    save_plot("freq_counts")


def plot_label_counts(label_stats):
    labels = list(label_stats.keys())
    counts = [label_stats[l]["count"] for l in labels]

    plt.figure()
    plt.bar(labels, counts)
    plt.title("Classifier Event Count per Label")
    plt.xlabel("Label")
    plt.ylabel("Count")
    save_plot("label_counts")


def plot_avg_duty_cycles(label_stats):
    labels = list(label_stats.keys())
    avg_duty = [label_stats[l]["avg_duty_cycle"] for l in labels]

    plt.figure()
    plt.bar(labels, avg_duty)
    plt.title("Avg Duty Cycle per Label")
    plt.xlabel("Label")
    plt.ylabel("Average Duty Cycle")
    save_plot("avg_duty_cycle")


def plot_avg_confidence(label_stats):
    labels = list(label_stats.keys())
    avg_conf = [label_stats[l]["avg_confidence"] for l in labels]

    plt.figure()
    plt.bar(labels, avg_conf)
    plt.title("Avg Confidence per Label")
    plt.xlabel("Label")
    plt.ylabel("Confidence")
    save_plot("avg_confidence")


def plot_snr(stats):
    snr = stats["snr_db"]
    plt.figure()
    plt.bar(["min", "avg", "max"], [snr["min"], snr["avg"], snr["max"]])
    plt.title("SNR (dB)")
    plt.ylabel("Value")
    save_plot("snr")


def plot_noise(stats):
    noise = stats["noise_floor_dbm"]
    plt.figure()
    plt.bar(["min", "avg", "max"], [noise["min"], noise["avg"], noise["max"]])
    plt.title("Noise Floor (dBm)")
    plt.ylabel("Value")
    save_plot("noise_floor")


def main():
    if len(sys.argv) < 3:
        print("Usage: python visualize.py schema.json stats.json")
        return

    schema_path = sys.argv[1]
    stats_path = sys.argv[2]

    # Load both JSONs
    schema = load_json(schema_path)
    stats = load_json(stats_path)

    # Prepare directory
    ensure_dir("plots")

    print("Saving plots to ./plots ...")

    plot_frequency_counts(stats["frequency_counts"])
    plot_label_counts(stats["label_stats"])
    plot_avg_duty_cycles(stats["label_stats"])
    plot_avg_confidence(stats["label_stats"])
    plot_snr(stats)
    plot_noise(stats)

    print("Done. Files saved under ./plots/")


if __name__ == "__main__":
    main()

