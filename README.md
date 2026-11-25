# ESC-50 ResNet-Tiny Reproduction and Improvements

A Digital Signal Processing (DSP) subject project that reproduces an environmental sound classification experiment and evaluates two proposed improvements.

## Paper Baseline

The baseline is the **ResNet-tiny** model reported by Goulão, Bandeira, Martins, and Oliveira:

> *Training environmental sound classification models for real-world deployment in edge devices*, Discover Applied Sciences (2024) 6:166.

The published ESC-50 5-fold cross-validation accuracy targets are:

| Model | Paper accuracy |
|---|---:|
| ResNet | 81.9% |
| ResNet-tiny | 81.6% |

This project reproduces both ResNet and ResNet-tiny, but the main comparison baseline is **ResNet-tiny**.

## Improvements

Two proposed approaches are evaluated against the reproduced baseline:

1. **P1 — Ensemble model:** Equal-probability ensemble of a COCO-pretrained ResNet-tiny and an ImageNet-pretrained EfficientNet-B0 using two log-mel spectrogram views.
2. **P2 — CNN–Transformer hybrid:** A pretrained ResNet-tiny CNN backbone followed by a lightweight Transformer encoder, with exponential moving average (EMA) weights and deterministic multi-view time-shift test-time augmentation (TTA).

Further implementation and experimental details are documented in `02-pembanding-new.ipynb`.

## Dataset and Protocol

- Dataset: ESC-50, 50 environmental sound classes
- Evaluation: official 5-fold cross-validation
- Each fold uses 4 folds for training and 1 fold for testing
- Input: 10-second audio represented as log-mel spectrogram data
- Training: 500 epochs per fold
- The final-epoch model is evaluated; no independent validation fold or test-fold tuning is used
- Augmentation: roll, frequency masking, time masking, and mixup

## Reproduced and Experimental Results

The reproduced ResNet-tiny baseline reached **85.85% mean 5-fold accuracy**, compared with **81.6%** reported in the paper.

| Method | Mean accuracy | Mean macro F1 | Accuracy improvement vs. baseline |
|---|---:|---:|---:|
| ResNet-tiny baseline | 85.85% | 85.33% | — |
| P1 ensemble | 89.05% | 88.57% | +3.20 percentage points |
| P2 CNN–Transformer + EMA + TTA | 87.00% | 86.45% | +1.15 percentage points |

Paired fold-level comparisons did not demonstrate statistical significance at $p < 0.05$:

| Comparison | Mean difference | Paired t-test p-value |
|---|---:|---:|
| P1 vs. baseline | +3.20 pp | 0.0645 |
| P2 vs. baseline | +1.15 pp | 0.0687 |

With only five official folds, the statistical power is limited. The improvements are therefore promising but should be interpreted as exploratory rather than statistically significant conclusions.

## Reproduction Steps

Run the notebooks in this order:

1. `01_reproduce.ipynb` — reproduce the ResNet-tiny and ResNet baselines.
2. `02-pembanding-new.ipynb` — train and evaluate P1 ensemble and P2 CNN–Transformer models.
3. `03-advanced-analysis.ipynb` — aggregate the saved fold results, compare models, and perform paired statistical tests.

The baseline notebook also clones the ESC-50 dataset and required repositories when they are not already available.

## Environment

Python and GPU-enabled PyTorch are recommended. Install the listed dependencies with:

```bash
pip install -r requirements.txt
```

The notebook setup can run locally or on Kaggle. On Kaggle, attach the saved outputs of the first two notebooks when running `03-advanced-analysis.ipynb`.

## Project Structure

```text
.
├── 01_reproduce.ipynb
├── 02-pembanding-new.ipynb
├── 03-advanced-analysis.ipynb
├── requirements.txt
├── esc50_repro/
│   ├── ESC-50/                 # ESC-50 dataset
│   └── yolov7/                 # pretrained backbone support
├── outputs/
│   ├── results/                # reproduced fold metrics and checkpoints
│   ├── figures/
│   └── summary_results.csv
├── outputs_advanced/
│   └── results/                # P1 and P2 fold metrics and checkpoints
└── outputs_03_advanced/
    ├── fold_metrics.csv
    ├── paired_tests.csv
    └── summary.csv
```

## Notes

- The P1 ensemble uses fixed equal probability voting and does not tune weights on test folds.
- P2 uses a lightweight Transformer, EMA, and fixed 3-view TTA rather than end-to-end learned fusion.
- Runtime and model size are included in the generated results to support edge-device trade-off analysis.
