# GPU Deep Learning Performance Lab

This project demonstrates a comprehensive approach to understanding and optimizing GPU performance for deep learning workloads, implementing the complete roadmap outlined in the initial plan.

## Project Structure
```
gpu-deep-learning-performance-lab/
├── benchmarks/
│   ├── resnet_training.py       # ResNet50 training benchmark
│   ├── vit_training.py          # Vision Transformer training benchmark  
│   └── bert_training.py         # BERT training benchmark
├── profiling/
│   ├── pytorch_profiler.py      # PyTorch profiler script
│   ├── pytorch_profiler_analysis.ipynb  # Profiler analysis notebook
│   └── gpu_utilization_report.ipynb     # GPU utilization report notebook
├── kernels/
│   ├── cuda_matmul.cu           # Basic CUDA matrix multiplication
│   ├── cuda_conv.cu             # CUDA convolution kernel
│   ├── cuda_matmul_benchmark.cu         # Benchmark for basic matmul
│   └── cuda_matmul_multisize_benchmark.cu # Multi-size matmul benchmark
├── experiments/
│   ├── fp32_vs_fp16_amp.py      # Mixed precision experiments
│   └── dataloader_benchmark.py  # Data loading optimization experiments
├── results/
│   ├── benchmarks/              # CSV results from all benchmarks
│   ├── plots/                   # Performance visualization plots
│   └── profiler_reports/        # Profiler output files
└── utils/
    ├── metrics.py               # Throughput tracking utilities
    └── plotting.py              # Plotting utilities
```

## Completed Work (4-Week Roadmap)

### Week 1 — GPU Fundamentals + Profiling ✓
- **Studied**: CUDA programming model, GPU memory hierarchy, tensor cores, kernel execution
- **Learned**: PyTorch profiler, Nsight Systems
- **Built**: `resnet_training_benchmark.py`
- **Measured**: GPU utilization, memory usage, training throughput
- **Deliverable**: Training benchmark report

### Week 2 — Mixed Precision & Optimization ✓
- **Implemented**: FP32 training, AMP (Automatic Mixed Precision) training
- **Compared**: Speed, GPU memory, convergence
- **Added experiment**: `fp32_vs_mixed_precision_experiment`
- **Deliverable**: Performance comparison plots

### Week 3 — CUDA Kernels ✓
- **Learned**: CUDA kernels, thread blocks, shared memory
- **Implemented kernels**: Matrix multiplication, convolution
- **Compared**: CUDA kernel vs PyTorch implementations
- **Deliverable**: Kernel benchmarking results

### Week 4 — Advanced Optimization ✓
- **Implemented**: Distributed training concepts, data loading optimization, kernel profiling
- **Added**: Performance report, optimization experiments
- **Deliverable**: Deep Learning GPU Performance Report (this README)

## Key Results

### Precision Benchmark (RTX 4060 Laptop GPU, ResNet50, batch size 64)
- **FP32 throughput**: 166.40 images/sec
- **AMP throughput**: 240.56 images/sec
- **FP32 peak memory**: 5672.04 MB
- **AMP peak memory**: 3159.98 MB

**AMP improved training throughput by ~44% while reducing peak GPU memory by ~44%.**

### Vision Transformer Benchmark (ViT-B/16, CIFAR-10)
| Batch Size | Throughput (images/sec) |
|------------|-------------------------|
| 16         | 47.31                   |
| 32         | 63.82                   |

### BERT Benchmark (BERT-base-uncased, SST-2 subset)
| Batch Size | Throughput (images/sec) |
|------------|-------------------------|
| 16         | 78.92                   |
| 32         | 93.17                   |
| 64         | 105.38                  |
| 128        | 9.73                    |

### CUDA Kernel Benchmark
- **Implemented**: Naive CUDA matrix multiplication for 512x512 matrices
- **Kernel runtime**: 0.737984 ms
- **Verified correctness** with C[0] = 512

### CUDA Kernel Scaling Analysis
Implemented naive and tiled shared-memory CUDA matrix multiplication kernels and benchmarked them across multiple matrix sizes:

| Matrix Size | Naive Time (ms) | Tiled Time (ms) | Speedup |
|-------------|-----------------|-----------------|---------|
| 256×256     | 0.0532          | 0.0790          | 0.67x   |
| 512×512     | 0.3661          | 0.2884          | 1.27x   |
| 1024×1024   | 2.8365          | 2.1803          | 1.30x   |
| 2048×2048   | 18.8211         | 12.8949         | 1.46x   |

**Observation**: Shared-memory tiling underperformed for small matrices due to synchronization and tiling overhead, but provided increasing speedups as matrix size grew. At 2048×2048, the tiled kernel achieved a 1.46× speedup over the naive baseline.

## What This Project Demonstrates

✅ **GPU compute understanding**: CUDA kernels for matrix multiplication and convolution
✅ **DL training optimization**: Mixed precision training showing significant speedups
✅ **Performance profiling**: PyTorch profiler analysis identifying bottlenecks
✅ **Kernel analysis**: Comparison of naive vs optimized CUDA implementations
✅ **Systems thinking**: Holistic view of deep learning performance from algorithms to hardware

## Skills Demonstrated
- **GPU Programming**: CUDA C/C++ kernel development and optimization
- **Deep Learning Frameworks**: PyTorch model training and profiling
- **Performance Engineering**: Systematic benchmarking and bottleneck identification
- **Memory Optimization**: Mixed precision training and memory usage analysis
- **Experimental Methodology**: Controlled experiments with measurable outcomes

## Files of Interest

1. **Benchmarks**: `benchmarks/*_training.py` - Training throughput measurements
2. **Experiments**: `experiments/fp32_vs_fp16_amp.py` - Mixed precision comparison
3. **Kernels**: `kernels/cuda_matmul*.cu` - CUDA implementations and benchmarks
4. **Profiling**: `profiling/*.ipynb` - Interactive profiler analysis
5. **Results**: `results/` - All benchmark data and visualizations

This project provides a complete demonstration of GPU performance analysis and optimization techniques that are directly applicable to machine learning systems engineering roles at companies like NVIDIA, Google, Meta, and other technology leaders in AI infrastructure.