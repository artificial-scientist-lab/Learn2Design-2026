# Evaluation environment

This reference describes the current evaluation setup, checked on 13 October
2026. Each submission uses a separate Python environment on Ubuntu 24.04.3 LTS.
The evaluation setup does not specify a common Docker base image.

See the [submission rules](submission.md) for hardware allocations, the time
budget, archive format, and permitted simulator access.

## Core versions

| Component | Version |
| --- | --- |
| Operating system | Ubuntu 24.04.3 LTS |
| Python | 3.12.3 |
| NVIDIA driver | 580.159.04 |
| JAX | 0.9.0.1 |
| JAXlib | 0.9.0.1 |
| jax-cuda13-plugin | 0.9.0.1 |
| jax-cuda13-pjrt | 0.9.0.1 |
| dfbench | 0.3.3 |
| differometor | 0.0.5 |

## Dependency policy

The evaluator fixes the simulator and JAX packages. Declarations for these
organizer-managed packages in `requirements.txt` are ignored:

```text
dfbench
differometor
jax
jaxlib
jax-cuda12-plugin
jax-cuda12-pjrt
jax-cuda13-plugin
jax-cuda13-pjrt
```

Other valid dependency declarations are honored when compatible with the fixed
core. This includes `nvidia-*` packages: their versions are not globally pinned,
and compatible pins in `requirements.txt` are honored. Requirements that cannot
be resolved together with the fixed core fail during environment preparation.
Ordinary package requirements are resolved from standard PyPI.

Each submission's environment is prepared before the timed optimization starts.
Matching the reference versions does not mean dependency installation is skipped.
Additional package versions can differ between submissions according to their
requirements.

## Reference NVIDIA library versions

The following versions were recorded in all seven Round 4 baseline environments.
They are a reproducible reference snapshot, rather than a guarantee that every
submission has identical NVIDIA library versions.

```text
nvidia-cublas==13.1.1.3
nvidia-cuda-crt==13.4.92
nvidia-cuda-cupti==13.0.85
nvidia-cuda-nvcc==13.4.92
nvidia-cuda-nvrtc==13.0.88
nvidia-cuda-runtime==13.0.96
nvidia-cudnn-cu13==9.24.0.43
nvidia-cufft==12.0.0.61
nvidia-cufile==1.15.1.6
nvidia-curand==10.4.0.35
nvidia-cusolver==12.0.4.66
nvidia-cusparse==12.6.3.3
nvidia-cusparselt-cu13==0.8.1
nvidia-nccl-cu13==2.30.7
nvidia-nvjitlink==13.4.92
nvidia-nvshmem-cu13==3.4.5
nvidia-nvtx==13.0.85
nvidia-nvvm==13.4.92
```
