import os
import pandas as pd
import matplotlib.pyplot as plt

def main():
    csv_path = "results/benchmarks/cuda_matmul_multisize_results.csv"
    output_dir = "results/plots"
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(csv_path)

    runtime_df = df.pivot(index="matrix_size", columns="kernel", values="time_ms")
    speedup_df = df[df["kernel"] == "tiled"][["matrix_size", "speedup_vs_naive"]]

    plt.figure(figsize=(8, 5))
    for col in runtime_df.columns:
        plt.plot(runtime_df.index, runtime_df[col], marker="o", label=col)
    plt.xlabel("Matrix Size")
    plt.ylabel("Execution Time (ms)")
    plt.title("CUDA MatMul Runtime vs Matrix Size")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cuda_matmul_runtime_vs_size.png"))
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(speedup_df["matrix_size"], speedup_df["speedup_vs_naive"], marker="o")
    plt.xlabel("Matrix Size")
    plt.ylabel("Speedup vs Naive")
    plt.title("Tiled CUDA MatMul Speedup vs Matrix Size")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cuda_matmul_speedup_vs_size.png"))
    plt.close()

    print("Saved plots to:", output_dir)

if __name__ == "__main__":
    main()