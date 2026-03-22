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
from transformers import BertForSequenceClassification, BertTokenizer
import datasets


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Device:", device)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    # Load a small dataset for quick training (e.g., SST-2 from GLUE)
    # We'll use the datasets library to load a subset of SST-2
    dataset = datasets.load_dataset("glue", "sst2", split="train[:1%]")  # Using 1% for speed

    # Load BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(examples["sentence"], padding="max_length", truncation=True, max_length=128)

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    # Safe batch sizes for ~8GB VRAM
    batch_sizes = [16, 32, 64, 128]

    results = []

    for batch_size in batch_sizes:

        print(f"Testing batch size: {batch_size}")

        # reset GPU memory
        torch.cuda.empty_cache()
        gc.collect()

        try:

            model = BertForSequenceClassification.from_pretrained("bert-base-uncased").to(device)

            loader = torch.utils.data.DataLoader(
                tokenized_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=True,
            )

            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

            tracker = ThroughputTracker()

            # warmup pass (important for GPU benchmarking)
            with torch.no_grad():
                dummy_input = {
                    "input_ids": torch.ones(batch_size, 128, dtype=torch.long, device=device),
                    "attention_mask": torch.ones(batch_size, 128, dtype=torch.long, device=device),
                }
                _ = model(**dummy_input)

            tracker.start()

            step = 0
            max_steps = 10

            for batch in loader:

                if step >= max_steps:
                    break

                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)

                optimizer.zero_grad()

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

                loss.backward()
                optimizer.step()

                tracker.update(input_ids.size(0))
                step += 1

            throughput, total_time = tracker.stop()

            print(f"Batch size {batch_size}")
            print(f"Total time: {total_time:.2f} sec")
            print(f"Throughput: {throughput:.2f} images/sec")

            results.append(
                {
                    "model": "bert-base-uncased",
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
                        "model": "bert-base-uncased",
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

    csv_path = os.path.join("results", "benchmarks", "bert_results.csv")

    with open(csv_path, mode="w", newline="") as csv_file:

        writer = csv.DictWriter(
            csv_file, fieldnames=["model", "batch_size", "precision", "throughput"]
        )

        writer.writeheader()
        writer.writerows(results)

    print("\nResults saved to:", csv_path)


if __name__ == "__main__":
    main()