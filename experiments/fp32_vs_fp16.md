# FP32 vs FP16 (AMP) Mixed Precision Experiment Results

## Overview
This experiment compares FP32 (full precision) and AMP (Automatic Mixed Precision) training for ResNet50 on CIFAR-10.

## Results

| Precision | Batch Size | Steps | Throughput (images/sec) | Peak Memory (MB) | Total Time (sec) |
|-----------|------------|-------|-------------------------|------------------|------------------|
| fp32      | 64         | 20    | 166.40                  | 5672.04          | 7.69             |
| amp       | 64         | 20    | 240.56                  | 3159.98          | 5.32             |

## Analysis

### Performance Improvement
- **Throughput Increase**: AMP shows ~44.6% higher throughput compared to FP32 (240.56 vs 166.40 images/sec)
- **Memory Reduction**: AMP uses ~44.3% less GPU memory (3159.98 MB vs 5672.04 MB)
- **Training Speed**: AMP completes training ~2.37 seconds faster per 20 steps

### Key Insights
1. Mixed precision training provides significant performance benefits on modern GPUs with Tensor Cores
2. Memory usage is substantially reduced, allowing for larger batch sizes
3. The speedup demonstrates effective utilization of tensor cores for matrix operations
4. Numerical stability is maintained with AMP through loss scaling

## Conclusion
AMP (Automatic Mixed Precision) provides clear advantages over FP32 training for ResNet50, offering both performance improvements and memory efficiency gains.