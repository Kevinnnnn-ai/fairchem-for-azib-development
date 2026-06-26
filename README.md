<div align="center">

# FAIR Chemistry for Aqueous Zinc-Ion Battery (AZIB) Development

Using Meta's FAIRChem UMA to test whether electrolyte additives can out-compete water for adsorption on zinc surfaces.

![Python](https://img.shields.io/badge/Python-3.11%E2%80%933.13-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.8.0%2Bcu128-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![FAIRChem](https://img.shields.io/badge/fairchem--core-2.21.0-0467DF?style=for-the-badge&logo=meta&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-12.8-76B900?style=for-the-badge&logo=nvidia&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![Build](https://img.shields.io/badge/Build-local-informational?style=for-the-badge)

</div>

---

## Ⅰ • Table of Contents

- [Ⅱ • Features](#ⅱ--features)
- [Ⅲ • Demonstration](#ⅲ--demonstration)
- [Ⅳ • Quick Start](#ⅳ--quick-start)
- [Ⅴ • Installation](#ⅴ--installation)
- [Ⅵ • Usage](#ⅵ--usage)
- [Ⅶ • Configuration](#ⅶ--configuration)
- [Ⅷ • Reference](#ⅷ--reference)
- [Ⅸ • License](#ⅸ--license)
- [Ⅹ • Authors](#ⅹ--authors)
- [Ⅺ • Contact](#ⅺ--contact)

<br>

## Ⅱ • Features

- **Competitive-adsorption screening** — computes ΔE_displace = E_ads(glycol) − E_ads(water) on each facet, the quantity that maps to whether an additive can displace interfacial water (poly(ethylene glycol)'s accepted "leveling agent" mechanism).
- **Deprotonation channel** — adds the deprotonated alkoxide (glycolate) with a dissociative ½H₂ reference, capturing the real Zn–O chemisorption that neutral glycol and water never show on a flat terrace.
- **Multiple Zn facets and defects** — flat Zn(002) basal (`hcp0001`) and Zn(100) prismatic (`hcp10m10`) terraces, plus engineered Zn(002)-adatom and Zn(002)-vacancy under-coordinated sites.
- **Machine-learned interatomic potential** — every relaxation is driven by FAIRChem's UMA checkpoint `uma-s-1p2` on a CUDA GPU, so there is no per-structure density-functional-theory cost.
- **Oxygen-down site scanning** — each adsorbate is oriented O-down and scanned over atop / bridge / hollow sites (or the single defect site); the strongest (most negative E_ads) is kept.
- **Reference-cancelling energetics** — adsorption energies share one calculator and gas-phase reference set, so systematic model error largely cancels in the reported ΔE values.
- **Contact classification** — labels every result chemisorbed / physisorbed / floating from the minimum anchor-oxygen-to-Zn distance.
- **Self-contained artifacts** — each run writes initial, optimized, and best geometries (`.xyz`), optimizer trajectories (`.traj`) and logs (`.log`), and a human-readable `summary.txt`.
- **Fast geometry tests** — a CPU-only test suite validates the geometry and energy-reference bookkeeping without ever loading the model.

<br>

## Ⅲ • Demonstration

A full screening run prints each facet's site-by-site adsorption energy, oxygen-to-zinc distance and surface gap, then writes a `summary.txt`. The committed `glycolate-defect-1` run produced:

```text
Hunting a chemisorbed state: glycol / water / glycolate on Zn facets + defects
E_ads(glycolate) is a DISSOCIATIVE energy referenced to 1/2 H2 (not a molecular
  adsorption energy): glycol(g) + slab -> glycolate(O-bound) + 1/2 H2(g).
dE_displace = E_ads(glycol) - E_ads(water)    ; <0 => glycol out-binds water
dE_deprot   = E_ads(glycolate) - E_ads(glycol); <0 => deprotonation deepens binding
Gas refs (eV): water=-14.152, glycol=-52.301, H2=-6.941

facet            E_ads(water) E_ads(glycol) E_ads(glycolate) dE_displace dE_deprot
Zn(002)                -0.055        -0.179            0.374      -0.124    +0.553
Zn(100)                -0.035        -0.197           -0.473      -0.162    -0.276
Zn(002)-adatom         -0.018        -0.155           -0.106      -0.137    +0.049
Zn(002)-vacancy         0.083        -0.245            0.180      -0.328    +0.425

Per-adsorbate best contact (min anchor-Zn distance):
  Zn(002)          water    : O-Zn = 3.97 Ang  (floating)
  Zn(002)          glycol   : O-Zn = 4.17 Ang  (floating)
  Zn(002)          glycolate: O-Zn = 2.07 Ang  (chemisorbed)
  Zn(100)          water    : O-Zn = 3.09 Ang  (physisorbed)
  Zn(100)          glycol   : O-Zn = 2.89 Ang  (physisorbed)
  Zn(100)          glycolate: O-Zn = 1.98 Ang  (chemisorbed)
  Zn(002)-adatom   water    : O-Zn = 3.52 Ang  (floating)
  Zn(002)-adatom   glycol   : O-Zn = 3.52 Ang  (floating)
  Zn(002)-adatom   glycolate: O-Zn = 1.84 Ang  (chemisorbed)
  Zn(002)-vacancy  water    : O-Zn = 4.46 Ang  (floating)
  Zn(002)-vacancy  glycol   : O-Zn = 5.56 Ang  (floating)
  Zn(002)-vacancy  glycolate: O-Zn = 2.03 Ang  (chemisorbed)

Note: neutral fragments, vacuum, no applied potential/explicit electrolyte.
Treat as relative mechanistic screening, not absolute interfacial energetics.
A chemisorbed (O-Zn < 2.4 Ang) glycolate or defect-site binding is the signal
that the neutral flat-terrace floating is lifted by deprotonation / under-coordination.
```

The takeaway: neutral water and glycol only float or physisorb, but the deprotonated glycolate forms a true Zn–O bond (O–Zn < 2.4 Å) on every facet.

Each individual relaxation also writes an Atomic Simulation Environment (ASE) `LBFGS` optimizer log, for example `data/logs/glycolate-defect-1/Zn100/opt_Zn100_glycolate_atop.log`:

```text
       Step     Time          Energy          fmax
LBFGS:    0 12:12:13      -74.489636        1.509158
LBFGS:    1 12:12:13      -74.611033        1.237466
LBFGS:    2 12:12:14      -74.750062        0.792540
LBFGS:    3 12:12:14      -74.794429        0.747417
LBFGS:    4 12:12:14      -74.972776        1.228634
...
```

All committed artifacts are sorted under [data/](data/) by type (`logs/`, `trajectories/`, `xyzs/`, `summaries/`) and then by run name.

<br>

## Ⅳ • Quick Start

```powershell
# 1. Clone and enter the repository
git clone https://github.com/Kevinnnnn-ai/fairchem-for-azib-development.git
cd fairchem-for-azib-development

# 2. Create and activate a virtual environment named .env.local
python -m venv .env.local
.\.env.local\Scripts\Activate.ps1

# 3. Install dependencies (pulls the CUDA 12.8 build of PyTorch)
pip install -r requirements.txt

# 4. Authenticate with Hugging Face so the gated UMA checkpoint can download
$env:HF_TOKEN = "hf_your_token_here"

# 5. Fast CPU-only sanity check (no GPU, no model download)
.\.env.local\Scripts\python.exe tests\test_geometry.py

# 6. Run the full screening (loads the UMA model on a CUDA GPU)
python src\main.py
```

**`Note`** — `python src\main.py` loads the UMA model on an NVIDIA CUDA GPU and relaxes dozens of structures, so it is long-running and downloads a multi-hundred-megabyte checkpoint on first use. It writes a fresh `stdout/runs/glycolate-defect-N/` each time and never overwrites previous runs.

<br>

## Ⅴ • Installation

### Requirements

- **Python 3.11–3.13** — `fairchem-core` requires `>=3.11,<3.14`; this environment uses 3.12.
- **An NVIDIA CUDA GPU with CUDA 12.8** — the code requests `device='cuda'`; RTX 50-series (Blackwell) GPUs specifically need the cu128 build.
- **A Hugging Face access token** — the UMA checkpoint is gated and must be downloaded with valid credentials.

### Dependencies

Pinned in [requirements.txt](requirements.txt). Only the two top-level packages are listed; their transitive dependencies (`ase`, `numpy`, `scipy`, and others) resolve automatically.

| Library | Version | Role |
|---------|---------|------|
| `torch` | `==2.8.0+cu128` | PyTorch tensor backend for the UMA model; the cu128 (CUDA 12.8) wheel is required for RTX 50-series GPUs |
| `fairchem-core` | `==2.21.0` | Meta FAIRChem; provides the pretrained UMA potential (`uma-s-1p2`) and the ASE-compatible `FAIRChemCalculator` |

`requirements.txt` also carries an `--extra-index-url https://download.pytorch.org/whl/cu128` line that points `pip` at PyTorch's CUDA wheel index for `torch`.

### Steps

```powershell
# 1. Clone the repository and move into it
git clone https://github.com/Kevinnnnn-ai/fairchem-for-azib-development.git
cd fairchem-for-azib-development

# 2. Create and activate a virtual environment named .env.local
python -m venv .env.local
.\.env.local\Scripts\Activate.ps1

# 3. Install all required dependencies
pip install -r requirements.txt
```

<br>

## Ⅵ • Usage

All commands run from the **repository root** with the project virtual environment active.

### Run the geometry tests

Fast, CPU-only checks of the pure geometry and energy-reference helpers; the model is never loaded:

```powershell
.\.env.local\Scripts\python.exe tests\test_geometry.py
```

### Smoke-test the calculator

Loads `uma-s-1p2` and relaxes a small Cu(100)+CO system to confirm the GPU and model stack work:

```powershell
python tests\quick_start.py
```

### Run the full screening

Builds the four facets, loads the UMA model, scans water / glycol / glycolate over every site, and writes results to `stdout/runs/glycolate-defect-N/`:

```powershell
python src\main.py
```

<br>

## Ⅶ • Configuration

There is no separate configuration file; the tunables are literals inside [src/main.py](src/main.py).

### Model and run scope

Set in `main()`:

| Setting | Default | Meaning |
|---------|---------|---------|
| model id | `'uma-s-1p2'` | FAIRChem UMA checkpoint loaded by `pretrained_mlip.get_predict_unit` |
| `device` | `'cuda'` | hardcoded GPU device for the predictor |
| `task_name` | `'oc20'` | FAIRChem task head used by `FAIRChemCalculator` |
| `FACETS` | `['Zn(002)', 'Zn(100)', 'Zn(002)-adatom', 'Zn(002)-vacancy']` | facets and defects screened in one run |

### Geometry and relaxation

Set in the helper functions:

| Setting | Default | Where | Meaning |
|---------|---------|-------|---------|
| `fmax`, `steps` | `0.05`, `300` | `relax()` | L-BFGS force tolerance (eV/Å) and maximum steps |
| `height` | `2.0` Å | `placeAdsorbate()` | initial anchor-oxygen height above the site |
| Zn(002) slab | `hcp0001('Zn', size=(3, 3, 4), vacuum=8.0)` | `buildFacet()` | basal-slab dimensions |
| Zn(100) slab | `hcp10m10('Zn', size=(3, 4, 4), vacuum=8.0)` | `buildFacet()` | prismatic-slab dimensions |
| contact thresholds | `< 2.4` chemisorbed, `≤ 3.2` physisorbed, else floating | `classify()` | O–Zn distance (Å) labels |
| run prefix | `'glycolate-defect'` | `getRunDir()` | output directory name under `stdout/runs/` |

The `ADSORBATES` dictionary defines each adsorbate (water, glycol, glycolate): its builder, anchor atom, orienting neighbours, and—for glycolate—the dissociative ½H₂ reference energy.

<br>

## Ⅷ • Reference

### Project layout

```text
fairchem-for-azib-development/
├─ src/
│  └─ main.py                 # the entire simulation: build -> relax -> score -> summarize
├─ tests/
│  ├─ test_geometry.py        # fast CPU-only geometry/reference tests (no model load)
│  └─ quick_start.py          # minimal FAIRChem GPU smoke test (Cu(100)+CO)
├─ data/                      # committed run artifacts, sorted by type then run
│  ├─ logs/<run>/...          # L-BFGS optimizer logs (.log)
│  ├─ trajectories/<run>/...  # ASE optimization trajectories (.traj)
│  ├─ xyzs/<run>/...          # init / final / best / reference geometries (.xyz)
│  └─ summaries/<run>/...     # per-run summary.txt
├─ docs/
│  ├─ personal/notes.md       # working notes
│  └─ superpowers/specs/      # design specs (e.g. glycolate-defect chemisorption)
├─ requirements.txt           # top-level pinned dependencies + cu128 wheel index
└─ .gitignore
```

Not under version control: `stdout/runs/` (where new runs write), `.env` (holds `HF_TOKEN`), and `.env.local/` (the virtual environment).

### Key entry points

- **`main()`** — the full screening pipeline: builds facets, loads the model, computes references, scans sites, and writes every artifact plus `summary.txt`.
- **`buildFacet(name)`** — returns `(slab, fixedIndices, slabTop, sites)` for `'Zn(002)'`, `'Zn(100)'`, `'Zn(002)-adatom'`, or `'Zn(002)-vacancy'`, fixing the bottom half of the layers.
- **`ADSORBATES`** — the registry of water, glycol, and glycolate, including how each is built, anchored, oriented, and referenced.
- **`relax(atoms, logPath, trajPath=None)`** — runs an ASE `LBFGS` relaxation with the shared FAIRChem calculator (`fmax=0.05`, `steps=300`).
- **`classify(oZn)`** — maps a minimum anchor-to-Zn distance to `chemisorbed` / `physisorbed` / `floating`.

### How it works

1. Build the three O-down adsorbates; glycolate is glycol with its O1 hydroxyl hydrogen removed.
2. Build each facet (flat Zn(002)/Zn(100) or Zn(002) with an adatom/vacancy), fix the bottom half of the layers, and expose adsorption sites.
3. Run a geometry-only pass first (no model) and write the initial placements, so geometry bugs fail fast.
4. Load the UMA calculator and compute gas-phase reference energies for water, glycol, and H₂ in 20 Å boxes.
5. Per facet: relax the clean slab, then scan each adsorbate over its sites, keeping the most negative E_ads = E(slab+ads) − E_slab − E_ref.
6. Report ΔE_displace and ΔE_deprot and classify each contact by minimum O–Zn distance.
7. Write every geometry, trajectory, and log, plus `summary.txt`, into a fresh `stdout/runs/glycolate-defect-N/`.

### External

- **Meta FAIRChem and the UMA models** — [github.com/facebookresearch/fairchem](https://github.com/facebookresearch/fairchem); the `uma-s-1p2` checkpoint is gated on Hugging Face and requires accepting Meta's model license and authenticating with a Hugging Face token (`HF_TOKEN`).
- **Atomic Simulation Environment (ASE)** — supplies the slab builders (`hcp0001`, `hcp10m10`), the `FixAtoms` constraint, and the `LBFGS` optimizer.

<br>

## Ⅸ • License

No license file is currently distributed with this project. Until a `LICENSE` is added, all rights are reserved by the author—please contact the maintainer before reuse or redistribution.

<br>

## Ⅹ • Authors

- **Kevinnnnn-ai** — author and maintainer ([github.com/Kevinnnnn-ai](https://github.com/Kevinnnnn-ai))

<br>

## Ⅺ • Contact

- **Repository** — [github.com/Kevinnnnn-ai/fairchem-for-azib-development](https://github.com/Kevinnnnn-ai/fairchem-for-azib-development)
- **Issues** — please open a [GitHub issue](https://github.com/Kevinnnnn-ai/fairchem-for-azib-development/issues) for bugs, questions, or feature requests

<br>

---

*Last Updated: June 22, 2026*
