import os
import pandas as pd
import matplotlib.pyplot as plt

def main():
    csv_path = "results/benchmarks/fp32_vs_amp_results.csv"
    output_dir = "results/plots"
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(csv_path)

    # Throughput plot
    plt.figure(figsize=(8, 5))
    plt.bar(df["precision"], df["throughput"])
    plt.xlabel("Precision Mode")
    plt.ylabel("Throughput (images/sec)")
    plt.title("ResNet50 Throughput: FP32 vs AMP")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "throughput_fp32_vs_amp.png"))
    plt.close()

    # Memory plot
    plt.figure(figsize=(8, 5))
    plt.bar(df["precision"], df["peak_memory_mb"])
    plt.xlabel("Precision Mode")
    plt.ylabel("Peak GPU Memory (MB)")
    plt.title("ResNet50 Peak Memory: FP32 vs AMP")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "memory_fp32_vs_amp.png"))
    plt.close()

    print("Saved plots to:", output_dir)

if __name__ == "__main__":
    main()