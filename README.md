# ML-Spec

Machine learning for extracting molecular electronic couplings from two-dimensional electronic spectra (2DES). Versioned public code for the Schultz Lab's ML spectroscopy papers.

## About this repository

ML-Spec is where the Schultz Lab (North Carolina Central University) publishes the code behind its papers on machine learning for multidimensional spectroscopy. Each paper cites the tagged release it used, so the results in any paper can be reproduced from the matching version of the code.

The repository is a fork of [Bridge-to-experiment-manuscript](https://github.com/jonathanschultzNU/Bridge-to-experiment-manuscript), the code for our 2025 Digital Discovery paper. The full history is kept on purpose: each later version can be traced back, commit by commit, to the code that produced the published 2025 results. The original repository is frozen as the record for that paper.

New methods are developed in the lab's private repository and released here no later than when the paper that uses them is submitted, so reviewers and readers can run them. Release history is in [CHANGELOG.md](CHANGELOG.md), and versioning, tagging and release rules are in [CONTRIBUTING.md](CONTRIBUTING.md).

### What's here now

The current code is the 2025 feed-forward neural network (FFNN), with small fixes made after release `1.0.0-alpha.3` (input-path handling and repository detection). Upcoming releases will restructure it into a modular package and extend it to new models; they will appear here with the papers they accompany.

## Releases and the papers they support

| Release | Paper | Code DOI |
|---|---|---|
| `1.0.0-alpha.3` | J. D. Schultz, K. A. Parker, B. Sbaiti and D. N. Beratan, "Using machine learning to map simulated noisy and laser-limited multidimensional spectra to molecular electronic couplings," *Digital Discovery*, 2025, 4, 1912–1924. [doi:10.1039/D5DD00125K](https://doi.org/10.1039/D5DD00125K) | [10.5281/zenodo.15499668](https://doi.org/10.5281/zenodo.15499668) (released from the original repository) |

To use the exact code from a paper, check out its tag, for example `git checkout 1.0.0-alpha.3`.

The data for the 2025 paper (`example_dataset.pkl` and `example_dataset_labels.csv`) are in the same Zenodo record.

---

## The 2025 FFNN code

### **Overview**
This package implements a **feed-forward neural network (FFNN)** for classifying electronic couplings in molecular dimers based on their **two-dimensional electronic spectra** (**2DES**).
The package supports:
- **Machine learning model training and evaluation**
- **Processing and augmentation of spectral datasets**
- **Visualization of spectra and ML results**
- **Dataset augmentation with experiment-informed modifications:**
  - **Noise addition and signal-to-noise filtering**
  - **Pump pulse resonance conditions (bandwidth and center frequency)**
- **Parameter scanning to observe how experimental conditions affect model performance**

### **Features**
#### **Machine Learning Pipeline**
- Custom PyTorch dataset class for spectral data
- Neural network training and evaluation
- Logging of performance metrics (accuracy, F1 scores)
- GPU support for accelerated training

#### **Data Processing**
- Loading and structuring 2D spectral data
- Classification of datasets based on electronic couplings
- User-defined preprocessing steps

#### **Data Augmentation**
- Additive (correlated or uncorrelated) and intensity-dependent noise injection
- Filtering of low-SNR spectra
- Pump modulation effects (bandwidth, center frequency)

#### **Visualization**
- Contour plots of 2D spectra
- Line plots for ML metrics (accuracy, loss, F1 scores)

---

## **Installation**
### **Using Conda (Recommended)**
First, create and activate a **conda environment** for the project:
```bash
conda env create -f environment.yml
conda activate bridge-to-exp-env
```
This ensures that all required dependencies are installed correctly.

### **Manual Installation with Pip**
Alternatively, install the dependencies manually:
```bash
pip install torch numpy pandas matplotlib scikit-learn gitpython
```

---

## **Usage**

### **Running the Code**
Write an input file (see below), then pass its path to `main.py`:
```bash
python main.py path/to/my_job.inp
```
The job name is taken from the input file name (`my_job.inp` gives `my_job`), and all outputs go to a folder of that name next to the input file. An example input is in [`examples/example_input.inp`](examples/example_input.inp). To run it, download `example_dataset.pkl` and `example_dataset_labels.csv` from the [Zenodo record](https://doi.org/10.5281/zenodo.15499668) into `examples/example_dataset/`, then run from the repository root, since the data paths in an input file are read relative to the folder you run from:
```bash
python main.py examples/example_input.inp
```

This will:
- Load the dataset
- Apply the specified augmentation (if any)
- Train the feed-forward neural network
- Iterate through the augmentation values (if any)
- Log accuracy and F1-score results

### **Directory Configuration**
The package needs the path to the spectral dataset and to its labels file (if using the files from Zenodo, these are `example_dataset.pkl` and `example_dataset_labels.csv`, respectively). Both paths must be specified in your input file.

### **Setting Up the Input File**
The input file specifies parameters for data processing and training, one `key = value` pair per line.

### **Input Parameters**
#### **Required Parameters**
| Parameter            | Description |
|----------------------|-------------|
| `database path`      | Folder that holds the spectral data. |
| `labels path`        | Path to the labels CSV (system parameters, including the electronic coupling). |
| `data loading method` | `single file` (one `.pkl` holding the whole dataset) or `iterative` (one file per system). |
| `data file name`     | Name of the `.pkl` file without the extension. Required for `single file` loading. |
| `task`              | Type of dataset modification (`noise`, `bandwidth`, `center frequency`, `bandwidth and center frequency`). |
| `class bounds`      | List of numerical boundaries defining classification bins for electronic coupling values. |
| `input size`        | Number of features in the dataset (e.g., flattened spectrum size). |
| `hidden layer size` | Number of neurons in the hidden layer of the FFNN. |
| `number of epochs`  | Number of training epochs. |
| `batch size`        | Number of samples per batch in training/testing. |
| `learning rate`     | Learning rate for the optimizer. |
| `dropout probability` | Dropout rate to prevent overfitting. |

#### **Task-Specific Parameters**
| Task Type  | Additional Required Parameters |
|------------|--------------------------------|
| `noise`    | `noise method` (`additive` or `intensity-dependent`), `noise fraction` (list of values to scan), `SNR filter`, `SNR threshold`. For additive noise, `noise submethod` (`correlated` or `uncorrelated`, default `uncorrelated`). |
| `bandwidth` | `bandwidth` (list of values to scan) and a single `center frequency` |
| `center frequency` | `center frequency` (list of values to scan) and a single `bandwidth` |
| `bandwidth and center frequency` | `bandwidth` and `center frequency` (both lists; every combination is run) |

#### **Optional Parameters**
| Parameter            | Default Value | Description |
|----------------------|--------------|-------------|
| `train-test split`  | `0.8` | Proportion of data allocated for training (the rest is for testing). |
| `save iteration outputs` | `"none"` | Whether to save ML reports per iteration. Options: `"none"`, `"partial"`, `"full"`. |
| `save training reports`  | `"false"` | Whether to save plots of training progress (accuracy, loss, F1 scores). |
| `save 2D plots`      | `"false"` | Whether to save 2D spectra during training. |
| `spec save interval` | `5` | Save the check spectra every N iterations. |
| `check-system ID`    | `11` | System whose spectra are saved for checking. |
| `t2 truncate`        | `"false"` | Whether to truncate `t2` time points. |
| `torch seed`         | `2942` | Random seed for PyTorch (ensures reproducibility). |
| `numpy seed`         | `72067` | Random seed for NumPy operations. |
| `split seed`         | `72067` | Random seed for train-test splitting. |

### **Example Input Files**

#### **Example 1: Noise Addition**
```bash
data loading method = single file
data file name = example_dataset
database path = ./examples/example_dataset/
labels path = ./examples/example_dataset/example_dataset_labels.csv
task = noise
noise method = intensity-dependent
noise fraction = 0, 0.01, 0.1, 1
SNR filter = True
SNR threshold = 0.01
class bounds = -805, -755, -705, -655, -605, -555, -505, -455, -405, -355, -305, -255, -205, -155, -105, -55, -5, 5, 55, 105, 155, 205, 255, 305, 355, 405, 455, 505, 555, 605, 665, 705, 755, 805
input size = 22801
hidden layer size = 100
number of epochs = 5
batch size = 100
learning rate = 0.01
dropout probability = 0.2
train-test split = 0.8
spec save interval = 1
```

#### **Example 2: Bandwidth and Center-Frequency Scanning**
```bash
data loading method = single file
data file name = example_dataset
database path = ./examples/example_dataset/
labels path = ./examples/example_dataset/example_dataset_labels.csv
task = bandwidth and center frequency
bandwidth = 10000, 2000, 1000
center frequency = 12000, 14500, 16500
class bounds = -805, -755, -705, -655, -605, -555, -505, -455, -405, -355, -305, -255, -205, -155, -105, -55, -5, 5, 55, 105, 155, 205, 255, 305, 355, 405, 455, 505, 555, 605, 665, 705, 755, 805
input size = 22801
hidden layer size = 100
number of epochs = 5
batch size = 100
learning rate = 0.01
dropout probability = 0.2
train-test split = 0.8
spec save interval = 1
```

### **Viewing Results**
Everything goes to `<job name>/`, next to the input file:

- `<job name>.mllog`: the run log, including the parameters used and the git commit of the code.
- `accuracies.csv` and `f1scores.csv`: performance metrics for each iteration.
- `inputs.pkl`: the parsed input parameters.

If requested, the following are also written:

- `check-spectra/`: 2D spectra of the check system (images and data), when `save 2D plots` is true.
- `training-reports/`: training-progress plots, when `save training reports` is true.
- `ml-reports/`: detailed per-iteration results. `full` saves training and testing results, `partial` testing only.

---

## Citing this work

If you use this code, please cite the paper for the release you used (see [Releases](#releases-and-the-papers-they-support)) and the code release itself. For the current code:

> J. D. Schultz, K. A. Parker, B. Sbaiti and D. N. Beratan, "Using machine learning to map simulated noisy and laser-limited multidimensional spectra to molecular electronic couplings," *Digital Discovery*, 2025, 4, 1912–1924. doi:10.1039/D5DD00125K

> Code: doi:10.5281/zenodo.15499668

## Acknowledgments

Development of this repository is assisted by Claude Code (Anthropic), an AI coding assistant. All changes are reviewed and committed by the maintainers.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
