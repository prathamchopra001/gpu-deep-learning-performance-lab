# experiments/fp32_vs_fp16_amp.py

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import csv
import gc
import time

import torch
import torchvision
import torchvision.transforms as transforms


def run_benchmark(precision_mode: str, batch_size: int = 64, max_steps: int = 20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
    ])

    dataset = torchvision.datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform,
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    model = torchvision.models.resnet50().to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    use_amp = precision_mode in {"fp16", "amp"}
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # Warmup
    with torch.no_grad():
        dummy = torch.randn(batch_size, 3, 224, 224, device=device)
        if precision_mode == "fp32":
            _ = model(dummy)
        else:
            with torch.cuda.amp.autocast():
                _ = model(dummy)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.time()

    total_images = 0
    steps = 0
    peak_mem_mb = 0.0

    for images, labels in loader:
        if steps >= max_steps:
            break

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if precision_mode == "fp32":
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        else:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        total_images += images.size(0)
        steps += 1

        if torch.cuda.is_available():
            peak_mem_mb = max(
                peak_mem_mb,
                torch.cuda.max_memory_allocated() / 1024**2
            )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    total_time = time.time() - start
    throughput = total_images / total_time if total_time > 0 else 0.0

    return {
        "precision": precision_mode,
        "batch_size": batch_size,
        "steps": steps,
        "throughput": round(throughput, 2),
        "peak_memory_mb": round(peak_mem_mb, 2),
        "total_time_sec": round(total_time, 2),
    }


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not available.")

    print("Device: cuda")
    print("GPU:", torch.cuda.get_device_name(0))
    print("Torch CUDA:", torch.version.cuda)

    os.makedirs("results/benchmarks", exist_ok=True)

    precision_modes = ["fp32", "amp"]
    # Pure fp16 training is usually less stable in standard training loops.
    # We'll keep it optional later, but AMP is the practical production baseline.

    results = []

    for mode in precision_modes:
        print(f"Running mode: {mode}")

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        try:
            result = run_benchmark(mode, batch_size=64, max_steps=20)
            results.append(result)
            print(result)

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"OOM in mode: {mode}")
                results.append({
                    "precision": mode,
                    "batch_size": 64,
                    "steps": 0,
                    "throughput": "OOM",
                    "peak_memory_mb": "OOM",
                    "total_time_sec": "OOM",
                })
            else:
                raise

    csv_path = "results/benchmarks/fp32_vs_amp_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "precision",
                "batch_size",
                "steps",
                "throughput",
                "peak_memory_mb",
                "total_time_sec",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    print("\nSaved results to:", csv_path)


if __name__ == "__main__":
    main()