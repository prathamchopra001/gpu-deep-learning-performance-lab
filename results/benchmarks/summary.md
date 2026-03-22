# Benchmark Summary

## Precision Benchmark (ResNet50, CIFAR10, RTX 4060 Laptop GPU)

| Precision | Batch Size | Steps | Throughput (img/s) | Peak Memory (MB) | Total Time (s) |
| --------- | ---------: | ----: | -----------------: | ---------------: | -------------: |
| FP32      |         64 |    20 |             176.29 |          5672.04 |           7.26 |
| AMP       |         64 |    20 |             285.77 |          3160.98 |           4.48 |

## CUDA Kernel Benchmark

| Kernel                      | Size    | Time (ms) | Output Check |
| --------------------------- | ------- | --------: | ------------ |
| Naive Matrix Multiplication | 512x512 |  0.737984 | C[0] = 512   |

## CUDA Kernel Benchmark

| Kernel            | Matrix Size | Time (ms) | Output Check |
| ----------------- | ----------- | --------: | ------------ |
| Naive CUDA MatMul | 1024x1024   |   2.83341 | C[0] = 1024  |
| Tiled CUDA MatMul | 1024x1024   |   2.18054 | C[0] = 1024  |

**Observed speedup:** 1.30x


## Multi-Size CUDA MatMul Benchmark

| Matrix Size | Naive Time (ms) | Tiled Time (ms) | Speedup |
| ----------: | --------------: | --------------: | ------: |
|         256 |          0.0532 |          0.0790 |   0.67x |
|         512 |          0.3661 |          0.2884 |   1.27x |
|        1024 |          2.8365 |          2.1803 |   1.30x |
|        2048 |         18.8211 |         12.8949 |   1.46x |

### Observation

Shared-memory tiling underperformed for small matrices due to synchronization and tiling overhead, but provided increasing speedups as matrix size grew. At 2048×2048, the tiled kernel achieved a 1.46× speedup over the naive baseline.
