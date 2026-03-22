import os
import pandas as pd
import matplotlib.pyplot as plt

def main():
    csv_path = "results/benchmarks/cuda_matmul_results.csv"
    output_dir = "results/plots"
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(csv_path)

    plt.figure(figsize=(8, 5))
    plt.bar(df["kernel"], df["time_ms"])
    plt.xlabel("Kernel")
    plt.ylabel("Execution Time (ms)")
    plt.title("CUDA Matrix Multiplication: Naive vs Tiled")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cuda_matmul_naive_vs_tiled.png"))
    plt.close()

    print("Saved plot to:", os.path.join(output_dir, "cuda_matmul_naive_vs_tiled.png"))

if __name__ == "__main__":
    main()