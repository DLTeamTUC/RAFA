# Data

The datasets used in the RAFA NoF 2026 experiments are **not distributed with this repository**.

Users must obtain the datasets from their official sources and place the required files under this `data/` directory before running the reproduction scripts.

## Expected directory structure

```text
data/
├── README.md
├── CICIoT2023.csv
└── CICDDoS2019/
    ├── MSSQL.csv
    └── DrDoS_DNS.csv
```

The raw dataset files are ignored by Git through the repository `.gitignore`.

## CICIoT2023

The main RAFA experiments use CICIoT2023.

The reproduction configuration expects the following local file:

```text
data/CICIoT2023.csv
```

The expected label column is:

```text
Label
```

The public code performs the preprocessing used by the paper artifact:

1. Load the CSV file.
2. Remove unusable feature columns according to the configured cleaning rules.
3. Convert the remaining features to numeric values.
4. Remove rows containing invalid or non-finite feature values.
5. Map labels to benign or attack classes.
6. Split the benign traffic into training, validation, test, and trusted-reference data.
7. Fit a `MinMaxScaler` on the benign training data only.
8. Apply the fitted scaler to the training, reference, validation, and test sets.
9. Use attack samples only for evaluation and attack construction, not for normal benign client training.

The default trusted benign reference-set size is:

```text
500
```

### Important note about the CICIoT2023 file

The paper-producing development environment used a consolidated file named:

```text
CICIoT2023.csv
```

If your downloaded CICIoT2023 distribution consists of multiple CSV files, you must first prepare an equivalent consolidated CSV before using the reproduction scripts.

The repository does not provide a dataset conversion script at this stage because the exact downloaded CICIoT2023 packaging can differ between distributions. The final consolidated file must retain the original feature columns and the `Label` column.

## CICDDoS2019

The cross-dataset validation uses selected CICDDoS2019 traffic subsets.

The reproduction configuration currently expects:

```text
data/CICDDoS2019/MSSQL.csv
data/CICDDoS2019/DrDoS_DNS.csv
```

The reproduction code scans the selected files, identifies their common numeric feature set, removes metadata/non-feature columns, and constructs independent benign training, trusted-reference, validation, and test splits for each selected source file.

The following columns are treated as non-feature metadata when present:

```text
Unnamed: 0
Flow ID
Source IP
Destination IP
Source Port
Destination Port
Timestamp
SimillarHTTP
```

The expected label column is:

```text
Label
```

Only the two selected source files above are required for the paper reproduction configuration currently included in this repository.

## Dataset placement check

From the repository root, the following files should exist before running the full experiments:

```text
RAFA/
├── data/
│   ├── CICIoT2023.csv
│   └── CICDDoS2019/
│       ├── MSSQL.csv
│       └── DrDoS_DNS.csv
├── configs/
├── src/
└── experiments/
```

You can then run the quick pipeline checks with:

```bash
python experiments/reproduce_ciciot2023.py --quick
```

and:

```bash
python experiments/reproduce_cicddos2019.py --quick
```

The `--quick` option is only a smoke test. It reduces the number of rounds and local batches and **does not reproduce the numerical results reported in the paper**.

For the full experiment configuration, run:

```bash
python experiments/reproduce_ciciot2023.py
```

and:

```bash
python experiments/reproduce_cicddos2019.py
```

## Data licensing

The original datasets remain subject to the terms, licenses, and citation requirements specified by their respective dataset providers.

This repository distributes only the RAFA source code, configuration files, experiment scripts, and reproduction documentation. It does not redistribute the original network-traffic datasets.
