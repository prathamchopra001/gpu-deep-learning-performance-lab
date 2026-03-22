import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torchvision
import torchvision.transforms as transforms
from utils.metrics import ThroughputTracker
import os
import csv
import gc


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Device:", device)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor()])

    dataset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )

    # Safe batch sizes for ~8GB VRAM
    batch_sizes = [16, 32, 64, 128, 256]

    results = []

    for batch_size in batch_sizes:

        print(f"Testing batch size: {batch_size}")

        # reset GPU memory
        torch.cuda.empty_cache()
        gc.collect()

        try:

            model = torchvision.models.resnet50().to(device)

            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=True,
            )

            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters())

            tracker = ThroughputTracker()

            # warmup pass (important for GPU benchmarking)
            with torch.no_grad():
                dummy = torch.randn(batch_size, 3, 224, 224).to(device)
                _ = model(dummy)

            tracker.start()

            step = 0
            max_steps = 10

            for images, labels in loader:

                if step >= max_steps:
                    break

                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                outputs = model(images)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                tracker.update(images.size(0))
                step += 1

            throughput, total_time = tracker.stop()

            print(f"Batch size {batch_size}")
            print(f"Total time: {total_time:.2f} sec")
            print(f"Throughput: {throughput:.2f} images/sec")

            results.append(
                {
                    "model": "resnet50",
                    "batch_size": batch_size,
                    "precision": "fp32",
                    "throughput": f"{throughput:.2f}",
                }
            )

        except RuntimeError as e:

            if "out of memory" in str(e).lower():

                print(f"OOM at batch size {batch_size}")

                results.append(
                    {
                        "model": "resnet50",
                        "batch_size": batch_size,
                        "precision": "fp32",
                        "throughput": "OOM",
                    }
                )

                # stop testing larger batches
                break

            else:
                raise e

    os.makedirs("results/benchmarks", exist_ok=True)

    csv_path = os.path.join("results", "benchmarks", "resnet_results.csv")

    with open(csv_path, mode="w", newline="") as csv_file:

        writer = csv.DictWriter(
            csv_file, fieldnames=["model", "batch_size", "precision", "throughput"]
        )

        writer.writeheader()
        writer.writerows(results)

    print("\nResults saved to:", csv_path)


if __name__ == "__main__":
    main()

