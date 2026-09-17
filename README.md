# RAFA

**Reconstruction-Aware Federated Aggregation for Unsupervised Reconstruction-Based Intrusion Detection Systems**

RAFA is a Byzantine-robust federated aggregation method for unsupervised reconstruction-based intrusion detection systems. It evaluates each client update by temporarily applying the update to the current global model and measuring its effect on a small trusted benign reference set. The resulting **Aggregation Quality Score (AQS)** is used to weight or reject updates that degrade trusted benign reconstruction behavior.

This repository accompanies the NoF 2026 paper:

> Beny Nugraha and Thomas Bauschert,  
> **“RAFA: Reconstruction-Aware Federated Aggregation for Unsupervised Reconstruction-Based Intrusion Detection Systems.”**

## Method overview

In a federated round, client \(i\) sends a model update \(\Delta \theta_i\) to the server. RAFA evaluates the candidate model obtained after applying this update to the current global model.

For a trusted benign reference set \(R\), let:

- \(\overline{RE}_0\) be the mean reconstruction error of the current global model on \(R\);
- \(\overline{RE}_i\) be the mean reconstruction error after applying client update \(i\).

RAFA uses the reconstruction-based score

\[
AQS_i =
\exp\left[
-\alpha
\max\left(
0,
\frac{\overline{RE}_i-\overline{RE}_0}
{\overline{RE}_0+\epsilon}
-m_{\mathrm{rec}}
\right)
\right].
\]

Updates with \(AQS_i < \tau\) are rejected. Accepted updates are aggregated using their AQS values as weights.

## Repository structure

```text
RAFA/
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── .gitignore
│
├── configs/
│   ├── paper_ciciot2023.yaml
│   ├── paper_cicddos2019.yaml
│   └── default_rafa.yaml
│
├── src/
│   ├── models.py
│   ├── data.py
│   ├── rafa.py
│   ├── aggregators.py
│   ├── attacks.py
│   ├── metrics.py
│   └── training.py
│
├── experiments/
│   ├── reproduce_ciciot2023.py
│   └── reproduce_cicddos2019.py
│
├── notebooks/
│   ├── RAFA_paper_reproduction.ipynb
│   └── archive/
│       └── RAFA_development.ipynb
│
├── results/
│   ├── tables/
│   └── figures/
│
└── data/
    └── README.md
```

The clean entry point for reproducing the paper is:

```text
notebooks/RAFA_paper_reproduction.ipynb
```

The original research and development notebook is retained under:

```text
notebooks/archive/RAFA_development.ipynb
```

It is preserved for provenance and may contain exploratory, diagnostic, or superseded experiment cells. It should not be treated as the primary reproduction interface.

## Installation

Python 3.10 or newer is recommended.

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it.

Linux/macOS:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Datasets

The datasets are **not distributed in this repository**.

The paper uses:

- **CICIoT2023** for the main evaluation;
- **CICDDoS2019** for additional cross-dataset validation.

Please obtain the datasets from their official sources and place the required files under the local `data/` directory. Do not commit the raw datasets to Git.

The repository will contain a separate `data/README.md` with the expected local directory structure.

## Reproducing the NoF 2026 results

The paper evaluates RAFA in the following settings:

| Paper result | Experiment |
|---|---|
| Reconstruction-targeted attacks | CICIoT2023 |
| Random-noise and sign-flip attacks | CICIoT2023 |
| Adaptive attacks | CICIoT2023 |
| Non-IID benign clients | CICIoT2023 |
| RAFA configuration sensitivity | CICIoT2023 |
| Reference-set-size sensitivity | CICIoT2023 |
| Cross-dataset validation | CICDDoS2019 |
| FedREDefense-inspired comparison | CICIoT2023 and CICDDoS2019 |

Start Jupyter with:

```bash
jupyter notebook
```

and execute:

```text
notebooks/RAFA_paper_reproduction.ipynb
```

from a fresh kernel.

## Main experimental configuration

The primary CICIoT2023 experiments use:

- 10 federated clients;
- 20 federated rounds;
- 5 configured local epochs;
- batch size 256;
- learning rate \(10^{-3}\);
- VAE latent dimension 16;
- trusted reference set size 500;
- \(\alpha = 15\);
- \(m_{\mathrm{rec}} = 0.01\);
- \(\tau = 0.2\);
- anomaly threshold set to the 95th percentile of benign validation reconstruction error.

### Important implementation detail: local batch cap

The paper-producing implementation additionally used a computational cap of:

```text
max_local_batches = 20
```

during the reported federated experiments. This repository preserves that setting in the paper reproduction configuration so that the released artifact reflects the code path used to generate the reported results.

## Reproducibility notes

### CICDDoS2019 configurations

The development experiments include source-specific RAFA settings for the CICDDoS2019 validation subsets. The paper reproduction configuration records these explicitly instead of presenting them as the universal RAFA defaults.

The nominal/default RAFA configuration remains:

```text
alpha = 15.0
m_rec = 0.01
tau = 0.2
```

### FedREDefense-inspired baseline

The repository contains a **FedREDefense-inspired adapted baseline** used for comparison in the paper. It is not intended to be a faithful reimplementation of the original FedREDefense method. In the development notebook, this comparison uses an update-space outlier proxy to represent the distinction between inspecting the update itself and RAFA's model-level reconstruction evaluation.

The code and documentation therefore label it as an **adapted/inspired baseline** rather than the original FedREDefense implementation.

### GPU reproducibility

Random seeds are controlled, but exact numerical equality across GPU architectures, PyTorch versions, and CUDA/cuDNN implementations is not guaranteed. Small run-to-run differences may occur.

## Metrics

The evaluation reports standard IDS metrics and RAFA-specific update-screening metrics:

- **F1 score**
- **AUC**: area under the ROC curve
- **FPR**: false positive rate
- **ASR**: attack success rate, measured as the fraction of attack samples below the anomaly-detection threshold
- **MDR**: malicious detection rate, i.e. the fraction of malicious client updates rejected by RAFA
- **BFRR**: benign false rejection rate, i.e. the fraction of honest client updates rejected by RAFA
- **AQS Gap**: difference between the mean AQS of benign and malicious updates

## Citation

If you use this code, please cite the corresponding NoF 2026 paper.

A machine-readable citation is provided in [`CITATION.cff`](CITATION.cff).

## License

This repository is released under the MIT License. See [`LICENSE`](LICENSE).

## Authors

- **Beny Nugraha**
- **Thomas Bauschert**

Chair of Communication Networks  
Chemnitz University of Technology  
Chemnitz, Germany

## Repository

https://github.com/DLTeamTUC/RAFA
