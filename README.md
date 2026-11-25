# Avant-Garde: Empowering GPUs with Scaled Numeric Formats

Simulation artifacts for the ISCA 2025 paper
[Avant-Garde: Empowering GPUs with Scaled Numeric Formats](https://doi.org/10.1145/3695053.3731100).
The repository holds the
[Accel-Sim](https://github.com/accel-sim/accel-sim-framework) configurations,
workloads and scripts used to compare block-scaled number formats (MX, HBFP)
on a conventional GPU against the same formats on Avant-Garde.

All experiments run on Accel-Sim in trace-driven mode with an NVIDIA A100
(SM80) model. Each number format is modeled by changing the Tensor Core
latency and initiation interval in `trace.config`.

## Repository layout

```
Avant-Garde/
├── Accel-Sim/
│   ├── configs/
│   │   ├── gpgpusim.config/   # GPU models (A100, A100 with power, RTX 3070, ...)
│   │   └── trace.config/      # Per-format Tensor Core latency settings
│   ├── scripts/               # Trace generation and simulation drivers
│   └── workloads/             # Microbenchmarks, custom WMMA kernels, DNN models
└── Accel-Sim-Dockerfile/
    ├── cuda11.7.1-cudnn8/     # Accel-Sim + PyTorch 2.0 + Transformers v4.21.1
    └── cuda11.8.0-cudnn8/     # Accel-Sim + NVBit tracer + PyTorch 2.0 + Transformers
```

### Configurations

`configs/gpgpusim.config/`

| Directory      | Description                                                       |
|----------------|-------------------------------------------------------------------|
| `A100`         | A100 baseline model with AccelWattch power simulation enabled     |
| `A100_energy`  | A100 model with power simulation disabled                         |
| `A100_lat6`    | A100 model used for the Tensor Core latency sweep                 |
| `A100_lat48`   | A100 model with a 100 cycle PTX Tensor Core latency               |
| `SM86_RTX3070` | RTX 3070 (SM86) model                                             |

`configs/trace.config/` sets the Tensor Core `latency,initiation` pair for
each format. Every other unit keeps the A100 default.

| Format  | Conventional GPU (`*_baseline`) | Avant-Garde (`*_Avant-Garde`) |
|---------|---------------------------------|-------------------------------|
| MXFP8   | 36, 24                          | 14, 9                         |
| MXINT8  | 30, 20                          | 8, 5                          |
| MX9     | 62, 41                          | 16, 12                        |
| MX4     | 59, 40                          | 7, 4                          |
| HBFP4   | 27, 18                          | 5, 3                          |

On the conventional GPU, scaling is done in software, so the baseline
latency includes the extra instructions needed to apply block scales
(12 extra instructions for 8 bit MX, 16 + 12 for 4 bit and MX9 formats).
Avant-Garde handles scaling inside the Tensor Core pipeline, so only a small
Tensor Core and pipeline latency is added.

Reference configurations without block scaling:

| Directory            | Tensor Core latency, initiation |
|----------------------|---------------------------------|
| `A100`               | 12, 8                           |
| `A100_FP16`          | 24, 16                          |
| `A100_FP8`           | 12, 8                           |
| `A100_INT8`          | 6, 4                            |
| `A100_Latency_6`     | 6, 8                            |
| `A100_Initiation_4`  | 12, 4                           |
| `A100_lat48`         | 48, 32                          |

### Workloads

| Directory            | Description                                                                 |
|----------------------|-----------------------------------------------------------------------------|
| `GPU_Microbenchmark` | Accel-Sim GPU microbenchmarks plus Tensor Core latency kernels (FP16/FP16, FP16/FP32, BF16/FP32, INT8/INT32) |
| `my_matmul`          | WMMA GEMM kernels. `tensor_core_scale.cu` applies per-block scales after each MMA to emulate software block scaling |
| `bert`               | BERT-base inference and training (Hugging Face, FP16 autocast)              |
| `gpt2`               | GPT-2 inference (Hugging Face, FP16 autocast)                               |
| `simple_DNN`         | Small PyTorch MLP and a single 1024 x 1024 matmul                           |

## Getting started

### 1. Build the Docker image

```bash
cd Accel-Sim-Dockerfile/cuda11.8.0-cudnn8
docker build -t avant-garde .
```

The `cuda11.8.0-cudnn8` image builds the NVBit tracer, which trace generation
requires. The `cuda11.7.1-cudnn8` image only builds the simulator.

### 2. Run the container

The scripts expect the following layout inside the container:

| Container path                  | Contents                          |
|---------------------------------|-----------------------------------|
| `/workspace/accel-sim-dev`      | Accel-Sim (built by the image)    |
| `/workspace/scripts`            | `Accel-Sim/scripts`               |
| `/workspace/workloads`          | `Accel-Sim/workloads`             |
| `/workspace/results`            | Traces and simulation outputs     |

```bash
docker run --gpus all -it \
  -v $(pwd)/Accel-Sim/scripts:/workspace/scripts \
  -v $(pwd)/Accel-Sim/workloads:/workspace/workloads \
  -v $(pwd)/Accel-Sim/results:/workspace/results \
  avant-garde bash
```

### 3. Install the configurations

Copy the configurations into the Accel-Sim `tested-cfgs` directories inside
the container:

```bash
cd /workspace/accel-sim-dev
cp -r /path/to/Accel-Sim/configs/gpgpusim.config/* gpu-simulator/gpgpu-sim/configs/tested-cfgs/
cp -r /path/to/Accel-Sim/configs/trace.config/*    gpu-simulator/configs/tested-cfgs/
```

`start_simulation.sh` reads both files from a directory with the same name,
so every `trace.config` directory used in a run needs a matching
`gpgpusim.config` directory (for example, copy `A100/gpgpusim.config` into
`A100_MXFP8_Avant-Garde/`).

### 4. Generate traces

```bash
cd /workspace/scripts
./generate_trace.sh <BENCHMARK> <BIN_FILE>
# e.g.
./generate_trace.sh bert_inference bert/inference_bert.py
```

- `BENCHMARK`: name of the trace directory
- `BIN_FILE`: binary or Python script, relative to `/workspace/workloads`

Traces are written to `/workspace/results/traces/<BENCHMARK>`. The script
traces kernels 9 to 43 by default (`DYNAMIC_KERNEL_LIMIT_START/END`), which
skips setup kernels in the PyTorch workloads.

### 5. Run simulations

```bash
./start_simulation.sh <BENCHMARK> <PLATFORM>
# e.g.
./start_simulation.sh bert_inference A100_MXFP8_Avant-Garde
```

Results are written to
`/workspace/results/simulation_results/<BENCHMARK>/<PLATFORM>.txt`.

### Microbenchmark sweep

Build the Tensor Core latency kernels under
`workloads/GPU_Microbenchmark/ubench/core/tensor_lat_half/` into
`GPU_Microbenchmark/bin/` as `fp16_fp16`, `fp16_fp32`, `bfloat16_fp32` and
`int8_int32`, then run:

```bash
./microbenchmark_gen_trace.sh   # trace all four kernels
./microbenchmark_simulate.sh    # simulate baseline vs Avant-Garde per format
```

Uncomment the lines in `microbenchmark_simulate.sh` for the formats you want
to simulate.

## Citation

```bibtex
@inproceedings{gil2025avantgarde,
  title     = {Avant-Garde: Empowering GPUs with Scaled Numeric Formats},
  author    = {Gil, Minseong and Ha, Dongho and Harma, Simla Burcu and Yoon, Myung Kuk and
               Falsafi, Babak and Ro, Won Woo and Oh, Yunho},
  booktitle = {Proceedings of the 52nd Annual International Symposium on Computer Architecture (ISCA)},
  pages     = {153--165},
  year      = {2025},
  publisher = {ACM},
  doi       = {10.1145/3695053.3731100}
}
```

## Acknowledgements

Built on [Accel-Sim](https://github.com/accel-sim/accel-sim-framework) and
its [GPU microbenchmark suite](https://github.com/accel-sim/gpu-app-collection).
