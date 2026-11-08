# Contributing to ML-Spec

ML-Spec is the public, versioned code for the Schultz Lab's machine-learning spectroscopy papers. Every release is tied to a paper, and a paper's results must be reproducible from the release it cites. These guidelines exist to keep that promise.

## What belongs here

- Code that a submitted or published paper relies on, released no later than when the paper is submitted so that reviewers and readers can run it.
- Fixes and documentation for code that is already released.
- Not unpublished methods, data, or results. Those stay in the lab's private development repository until their paper is submitted.

## Branches and pull requests

- `main` is always the latest released code plus reviewed changes. Don't commit to it directly; work on a branch and open a pull request.
- Name branches by what they do, for example `fix/input-path` or `docs/release-checklist`.
- Check the base repository on every pull request. ML-Spec is a fork, and GitHub's default base for a new pull request is the parent repository (`jonathanschultzNU/Bridge-to-experiment-manuscript`), which is frozen. The base must be `NCCU-Schultz-Lab/ML-Spec`, branch `main`.
- Keep each pull request to one purpose, and add a line to the `Unreleased` section of [CHANGELOG.md](CHANGELOG.md) in the same pull request.
- The maintainer (Jonathan Schultz) reviews and merges all pull requests.

## Commit messages

- A summary line in the imperative mood, about 72 characters or fewer: "Fix output folder for relative input paths", not "Fixed stuff".
- A blank line, then a body that explains why, when the reason isn't obvious.
- Commits written with an AI coding assistant end with a trailer naming it, for example:

  ```
  Co-Authored-By: Claude <noreply@anthropic.com>
  ```

## Versioning

ML-Spec follows [Semantic Versioning 2.0.0](https://semver.org/): `MAJOR.MINOR.PATCH`.

- MAJOR: changes that break existing inputs, configs or outputs (for example, replacing `.inp` input files with a new config format).
- MINOR: new features that leave existing inputs working (a new model, a new augmentation).
- PATCH: bug fixes and documentation.
- Pre-releases: `2.0.0-alpha.1`, `2.0.0-beta.1`, `2.0.0-rc.1`. Use them for checkpoints while a release is being prepared.

## Tags

- Tag names are the bare version with no `v` prefix (`2.0.0`, `2.0.0-alpha.1`), matching the existing tags.
- Use annotated tags, which record who tagged, when, and a message:

  ```bash
  git tag -a 2.0.0 -m "ML-Spec 2.0.0: <one-line summary>"
  git push origin 2.0.0
  ```

- Never move or delete a tag once it's pushed. Zenodo archives and paper citations point at tags; a moved tag would make a citation point at different code. A mistake gets a new version number instead. A GitHub tag ruleset blocks deletion and force-updates of tags.
- Tag only commits on `main`.

### Tags from before the fork

`1.0.0-alpha.1`, `1.0.0-alpha.2` and `1.0.0-alpha.3` were created in the original repository and carried over unchanged, so the published code can be checked out and compared within ML-Spec. They are tags only: they have no GitHub Releases here, because publishing a Release would make Zenodo archive the same code under a second DOI. Their Zenodo records belong to the original repository. `1.0.0-alpha.1` and `1.0.0-alpha.2` point at the same commit.

## Releasing

1. Confirm the release's results reproduce: every claim in the accompanying paper can be regenerated from this code and the published data.
2. In [CHANGELOG.md](CHANGELOG.md), rename `Unreleased` to the new version with today's date (`## [2.0.0] - YYYY-MM-DD`), start a new empty `Unreleased` section, and update the comparison links at the bottom.
3. Update the release metadata: in `CITATION.cff`, the `preferred-citation` (the paper this release supports) and the authors; in `.zenodo.json`, the creators and the `isSupplementTo` paper DOI; and the version in the package metadata, once there is one. Zenodo takes the version number from the tag.
4. Merge those changes to `main` through a pull request.
5. Create an annotated tag on that commit and push it (see Tags).
6. On GitHub, publish a Release from the tag, using the changelog section as its notes. Zenodo archives the Release and assigns a version DOI.
7. Add the release, its paper and its version DOI to the "Releases and the papers they support" table in [README.md](README.md), and cite the version DOI in the paper.

## Changelog

[CHANGELOG.md](CHANGELOG.md) follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Entries go under `Added`, `Changed`, `Deprecated`, `Removed`, `Fixed` or `Security`, written for a reader of the code, not as a copy of commit messages.

## Reproducibility

- Fix random seeds through the input parameters, and keep their defaults stable within a major version.
- The run log records the git commit of the code that produced it; keep that working.
- Results reported in a paper must be reproducible from the data published with it, which may be a subset of the full training data.

## AI assistance

Development is assisted by Claude Code (Anthropic). Every change is reviewed by a person before it's committed; AI-assisted commits carry a `Co-Authored-By` trailer, and the README acknowledges the assistance.
