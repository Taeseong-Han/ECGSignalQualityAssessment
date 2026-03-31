# Reproducing the Paper

This document provides step-by-step instructions to reproduce the full training and evaluation pipeline from the paper.

For **installation and usage**, see [README.md](README.md).

[![arXiv](https://img.shields.io/badge/arXiv-2506.11815-b31b1b.svg)](https://arxiv.org/abs/2506.11815)

---

## 🧬 Data Preprocessing and Training

#### 🖼️ Framework Overview

![Framework Overview](figures/Framework_overview.jpg)

---

### 🔄 Superlet Transform on PTB-XL Data

To convert raw PTB-XL ECG recordings into superlet scalograms, run the following command from the project root:

```bash
python -m preprocessing.superlet_transform_ptbxl \
  --ptbxl_raw_path [PTBXL_RAW_PATH]
```

Replace [PTBXL_RAW_PATH] with the full path to your downloaded PTB-XL dataset , e.g.:

```bash
--ptbxl_raw_path ~/Database/physionet.org/files/ptb-xl/1.0.3
```

📥 **Download the dataset from PhysioNet:**

🔗 https://physionet.org/content/ptb-xl/1.0.3/

<br>

### 🏋️Train Model

Train the autoencoder on discretized superlet scalograms:

```bash
python -m train.train_autoencoder \
  --discretization \
  --save_path [YOUR MODEL PATH]
```

This model compresses input scalograms into a latent space and is a required pretraining step for training the latent
diffusion model.
If you're using the vanilla diffusion model, you can train it directly without this autoencoder stage.
You can customize training using additional CLI options.

> ⚠️ The training script structure is unified across this project.  
> The same CLI pattern applies to training:
>
> - `train_autoencoder`
> - `train_latent_diffusion`
> - `train_vanilla_diffusion`
>
> Simply execute:
>
> ```bash
> python -m train.[script_name] [options]
> ```

<br>

### 📊 Noise Level Quantification

Quantify ECG signal noise using a pretrained diffusion model.
Reconstruction metrics like PSNR serve as proxies for noise severity.

This outputs per-lead metrics (PSNR, SSIM, etc.) in CSV format for further analysis.

```bash
python -m evaluation.run_noise_quantification \
  --checkpoint [YOUR MODEL PATH] \
  --timestep 250 \
  --noise_scheduler_type ddim \
  --step_interval 10 \
  --discretization \
  --output_dir ./evaluation/results
```

To evaluate all 10 folds (not just the test set), include:

```bash
--include_all_folds
```

<br>

### 📈 Performance Evaluation Across Experiments

Compare multiple model configurations by computing Wasserstein-1 distances (W₁) between metric distributions:

```bash
python -m evaluation.eval_models \
  --keyword ddpm \
  --output_dir ./evaluation/results
```

**Arguments:**

- --keyword: Substring used to filter result files (e.g., 'ddpm', 't250', etc.)

This helps quantify how well different models separate clean and noisy segments under various noise types (static,
burst, baseline).

<br>

### 🧹 Dataset Refinement and Retraining

Improve training by filtering clean segments based on model, not human labels.

#### Step 1. Quantify Noise Across All Folds

Run noise quantification over the full PTB-XL dataset, including clean-labeled segments:

```bash
python -m evaluation.run_noise_quantification \
  --checkpoint  [YOUR MODEL PATH] \
  --timestep 250 \
  --noise_scheduler_type ddpm \
  --include_all_folds \
  --discretization \
  --output_dir ./evaluation/results

```

This provides reconstruction-based quality scores (e.g., PSNR) for every segment in the dataset.

#### Step 2: Select High-Confidence Clean Segments and Retrain

Identify top-N% of clean segments with the highest quality (e.g., PSNR)
under the model that showed strong sensitivity to static and burst noise (high W₁-distance):

```bash
python -m train.retrain_autoencoder \
  --static_file ./evaluation/results/dm_ddpm_t250_psnr.csv \
  --burst_file ./evaluation/results/ldm_ddpm_t50_psnr.csv \
  --metric psnr \
  --static_percentage 0.5 \
  --burst_percentage 0.5 \
  --discretization \
  --save_path ./output/ae_model_refined
```

This filters out mislabeled or ambiguous clean segments and enables retraining on a refined,
high-confidence subset of the original dataset.

<br>

---

## 📄 License and Citation

The software is licensed under the MIT License 2.0.  
Please cite the following paper if you use this code:

```bibtex
@misc{han2025diffusionbasedelectrocardiographynoisequantification,
  title={Diffusion-Based Electrocardiography Noise Quantification via Anomaly Detection}, 
  author={Tae-Seong Han and Jae-Wook Heo and Hakseung Kim and Cheol-Hui Lee and Hyub Huh and Eue-Keun Choi and Dong-Joo Kim},
  year={2025},
  eprint={2506.11815},
  archivePrefix={arXiv},
  primaryClass={eess.SP},
  url={https://arxiv.org/abs/2506.11815}
}
```