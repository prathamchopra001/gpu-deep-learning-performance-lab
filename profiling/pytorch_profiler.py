import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.profiler import profile, record_function, ProfilerActivity


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not available.")

    device = torch.device("cuda")
    print("GPU:", torch.cuda.get_device_name(0))

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
        batch_size=64,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    model = torchvision.models.resnet50().to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    os.makedirs("results/profiler_reports", exist_ok=True)

    data_iter = iter(loader)
    images, labels = next(data_iter)
    images = images.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        for step in range(5):
            with record_function("resnet_train_step"):
                optimizer.zero_grad(set_to_none=True)
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

    table = prof.key_averages().table(
        sort_by="cuda_time_total",
        row_limit=20
    )

    print("\nTop CUDA ops:\n")
    print(table)

    output_path = "results/profiler_reports/resnet_profile.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(table)

    print("\nSaved profiler report to:", output_path)


if __name__ == "__main__":
    main()