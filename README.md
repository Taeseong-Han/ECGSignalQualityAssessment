# ECG Signal Quality Assessment

Diffusion-based ECG signal quality assessment using reconstruction error (PSNR) as a noise proxy — no noise labels required.

[![arXiv](https://img.shields.io/badge/arXiv-2506.11815-b31b1b.svg)](https://arxiv.org/abs/2506.11815)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-Pretrained%20Model-blue)](https://huggingface.co/Taeseong-Han/ECGNoiseQuantification)

---

## Installation

**1. Clone the repository**

```bash
git clone https://github.com/Taeseong-Han/ECGSignalQualityAssessment.git
cd ECGSignalQualityAssessment
```

**2. Install PyTorch and JAX for your CUDA version**

Install PyTorch and JAX separately to match your CUDA environment.
Visit the official installation guides:

- PyTorch: https://pytorch.org/get-started/locally/
- JAX: https://jax.readthedocs.io/en/latest/installation.html

**3. Install remaining dependencies**

```bash
pip install -r requirements.txt
```

**4. Download the pretrained model**

```bash
pip install huggingface_hub
huggingface-cli download Taeseong-Han/ECGNoiseQuantification pretrained_ldm.pth --local-dir .
```

Or download manually from [HuggingFace](https://huggingface.co/Taeseong-Han/ECGNoiseQuantification/blob/main/pretrained_ldm.pth).

---

## Quick Start

```python
import numpy as np
from utils.inference import ecg_noise_quantification

ecg = np.load("./data/database/ecg_example.npz")['signal']  # shape: (5000,)

output = ecg_noise_quantification(
    ecg=ecg,
    sampling_freq=500,
    checkpoint_path="./pretrained_ldm.pth",
)

print(output.psnr)  # shape: (1, 1) — higher PSNR = cleaner signal
```

See [demo.ipynb](demo.ipynb) for full examples with visualizations.

---

## Interpreting PSNR as Signal Quality

PSNR (Peak Signal-to-Noise Ratio) serves as a continuous signal quality metric.
Higher values indicate cleaner signals. The thresholds below are empirically validated on external datasets (BUT QDB, CinC Challenge 2011):

| PSNR Range | Interpretation |
|:----------:|:--------------:|
| **< 24**   | Heavily noisy  |
| **24 – 26** | Locally / partially noisy |
| **> 26**   | Likely clean   |

![PSNR-based signal quality validation](figures/external_validation.jpg)

The figure above shows that PSNR-based assessment can reveal labeling inconsistencies in human-annotated datasets, where segments labeled "clean" sometimes exhibit clear noise artifacts and vice versa.

---

## Usage Examples

### Single-lead, Global PSNR

Returns one PSNR value per 10-second segment:

```python
output = ecg_noise_quantification(
    ecg=ecg,                        # (samples,) or (leads, samples)
    sampling_freq=500,
    checkpoint_path="./pretrained_ldm.pth",
    n_partitions=1,                 # global PSNR (default)
)
# output.psnr shape: (1, num_segments)
```

### Single-lead, Local PSNR

Subdivide each segment into finer partitions for localized quality analysis:

```python
output = ecg_noise_quantification(
    ecg=ecg,
    sampling_freq=500,
    checkpoint_path="./pretrained_ldm.pth",
    n_partitions=8,                 # 8 partitions per segment
)
# output.psnr shape: (1, total_valid_partitions)
```

### Multi-lead ECG

Works with any number of leads (e.g., 12-lead ECG):

```python
ecg_12lead = np.random.randn(12, 12000)  # 12 leads, 24 seconds

output = ecg_noise_quantification(
    ecg=ecg_12lead,
    sampling_freq=500,
    checkpoint_path="./pretrained_ldm.pth",
    n_partitions=1,
)
# output.psnr shape: (12, 3)  — 12 leads × 3 segments
```

### Key Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ecg` | `np.ndarray` | — | Input ECG signal. Shape: `(samples,)` or `(leads, samples)` |
| `sampling_freq` | `int` | — | Sampling frequency in Hz |
| `checkpoint_path` | `str` | — | Path to pretrained model weights |
| `n_partitions` | `int` | `1` | Number of partitions per segment (`1` = global PSNR) |
| `return_images` | `bool` | `False` | If `True`, returns original and cleaned scalograms |
| `batch_size` | `int` | `64` | Batch size for inference |

See [demo.ipynb](demo.ipynb) for complete examples including scalogram visualizations and MSE maps.

---

## Citation

```bibtex
@misc{han2025labelnoise,
  title={Label-Noise Resilient ECG Signal Quality Assessment via Diffusion Models and Wasserstein-Guided Data Refinement},
  author={Tae-Seong Han and Jae-Wook Heo and Hakseung Kim and Cheol-Hui Lee and Hyub Huh and Eue-Keun Choi and Dong-Joo Kim},
  year={2025},
  eprint={2506.11815},
  archivePrefix={arXiv},
  primaryClass={eess.SP},
  url={https://arxiv.org/abs/2506.11815}
}
```

---

## Reproducing the Paper

For the full training and evaluation pipeline (data preprocessing, model training, dataset refinement), see [REPRODUCE.md](REPRODUCE.md).

---

## License

This project is licensed under the [MIT License](LICENSE).
