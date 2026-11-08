# Changelog

All notable changes to ML-Spec are recorded here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow [Semantic Versioning](https://semver.org/). Releasing and tagging rules are in [CONTRIBUTING.md](CONTRIBUTING.md).

## [Unreleased]

### Added

- `LICENSE` (MIT).
- `CONTRIBUTING.md` (branches, commit messages, versioning, tagging and the release checklist) and this changelog.
- `CITATION.cff`, for GitHub's "Cite this repository" button, and `.zenodo.json`, the metadata Zenodo uses when it archives a release (MIT license, creators, and links to the 2025 code and paper).
- A pull request template with the review and release checklists.
- `.gitignore` for Python caches, virtual environments, build artifacts and local data, including the Zenodo example data downloaded into `examples/example_dataset/`.

### Changed

- The repository is now ML-Spec, a fork of `Bridge-to-experiment-manuscript` in the NCCU-Schultz-Lab organization. The README explains its purpose, maps releases to papers, and gives the citation.
- The input path given to `main.py` is resolved to an absolute path, and the job name is taken from its file name.
- README usage docs now match the code: the run command, the required data parameters, `noise submethod`, the pump-scan parameters and the output folder names.

### Fixed

- The run log records the code's git commit even when a job is started from outside the repository (`get_git_info` now locates the repository from the code's own location).
- The README names the Zenodo data files correctly (`example_dataset.pkl`, `example_dataset_labels.csv`).

## [1.0.0-alpha.3] - 2025-05-23

The code for the published paper: Schultz, Parker, Sbaiti and Beratan, *Digital Discovery*, 2025, 4, 1912–1924 ([doi:10.1039/D5DD00125K](https://doi.org/10.1039/D5DD00125K)). Archived from the original repository at [doi:10.5281/zenodo.15499668](https://doi.org/10.5281/zenodo.15499668). Changes made during peer review:

### Added

- Correlated additive noise (constant along the detection axis) alongside uncorrelated additive noise, chosen with the `noise submethod` parameter.
- An example input file, `examples/example_input.inp`.
- The `data loading method` parameter (`single file` or `iterative`).
- A debugging tools module.

### Changed

- Dataset and labels paths are set in the input file (`database path`, `labels path`) instead of `config.py`, which was removed.
- The SNR filter uses the measured mean absolute noise as the noise width.
- The training loop moved from `modules/training_pipeline.py` into `main.py`.

## [1.0.0-alpha.2] - 2025-03-17

Points at the same commit as `1.0.0-alpha.1`; no code changes. Archived on Zenodo from the original repository.

## [1.0.0-alpha.1] - 2025-03-17

First public release, from the original repository: a feed-forward neural network that classifies the electronic coupling of molecular dimers from their two-dimensional electronic spectra, with additive and intensity-dependent noise, pump bandwidth and center-frequency effects, an SNR filter, and parameter scans.

[Unreleased]: https://github.com/NCCU-Schultz-Lab/ML-Spec/compare/1.0.0-alpha.3...HEAD
[1.0.0-alpha.3]: https://github.com/NCCU-Schultz-Lab/ML-Spec/compare/1.0.0-alpha.2...1.0.0-alpha.3
[1.0.0-alpha.2]: https://github.com/NCCU-Schultz-Lab/ML-Spec/tree/1.0.0-alpha.2
[1.0.0-alpha.1]: https://github.com/NCCU-Schultz-Lab/ML-Spec/tree/1.0.0-alpha.1
