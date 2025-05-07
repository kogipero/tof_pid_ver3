# Barrel TOF PID Analysis

This repository contains a **PID‑performance evaluation pipeline** for the *Barrel Time‑Of‑Flight* (BTOF) detector of the ePIC experiment.
It is designed to post‑process **EICrecon** output files (EDM4hep ROOT) produced from DD4hep Monte Carlo simulation, apply several matching steps and finally evaluate π/K/p identification performance.

---

## Highlights

* **Single YAML configuration** toggles input file, TTree/branch names and basic selection cuts.
* Fully vectorised analysis with **uproot + awkward + numpy** – no ROOT I/O on the Python side.
* Automatic ROOT plots (TH1/TH2/TGraph/TCanvas)

  * β<sup>−1</sup> vs p
  * reconstructed mass spectra
  * PID efficiency / purity / Gaussian separation vs *p*<sub>T</sub>
* All results are written to `out/<directory_name>/`

  * analysis ROOT file (`<output>.root`)
  * optional intermediate CSV files (MC⇔TOF⇔Track matches)
    *(delete them afterwards if disk space matters)*

---

## Processing Steps (overview)

```
MC truth  →  EICrecon  →  TOF‑MC matching  →  TOF‑Track matching  →  PID metrics  →  ROOT / CSV
```

---

## Tested Runtime Environment

| Component | Version | Notes (if any)        |
| --------- | ------- | --------------------- |
| Python    | 3.10.9  | Miniconda recommended |
| ROOT      | 6.32/02 | PyROOT enabled        |
| uproot    | 5.3.10  |                       |
| awkward   | 2.6.5   |                       |
| numpy     | 1.26.4  |                       |
| PyYAML    | 6.0.1   |                       |
| tqdm      | 4.66.4  |                       |

> *Please create an issue or PR if another required package / version is missing.*

---

## Directory Layout

```
.
├── src/
│   ├── analyzer_base.py
│   ├── mc_analyzer.py
│   ├── tof_analyzer.py
│   ├── track_analyzer.py
│   ├── matching_mc_and_tof.py
│   ├── matching_tof_and_track.py
│   ├── tof_pid_performance_manager.py
│   ├── pipeline_utils.py
│   ├── utility_functions.py
│   ├── XXX_plotter.py
│   └── helper_functions.py
├── config/
│   └── config.yaml              # create one per dataset type
├── analyze_script.py            # entry‑point
└── out/                         # results (git‑ignored)
```

---

## Example YAML (`config/config.yaml`)

```yaml
analysis:
  directory_name: eic_pid_test
  analysis_event_type: NCDIS        # will be overridden by --filetype
  selected_events: 10000
  verbose: true                     # (some flags still WIP)
  plot_verbose: true                # ↳ same here
  detail_plot_verbose: false
  version: "ver1_24_2"

vertex_cuts:                        # not implemented yet (hard‑coded)
  zvtx_min: -100.0   # [mm]
  zvtx_max:  100.0

file_paths:
  ncd1:
    description: NCDIS
    path: /path/to/pythia8NCDIS_18x275.edm4hep.root
  sp_pion:
    description: single_particle_pion
    path: /path/to/pion_4GeV.edm4hep.root

branches:
  mc:
    mc_pdg:      events/MCParticles.PDG
    mc_vertex_x: events/MCParticles.vertex.x
    # ... add more as needed

  tof:
    tof_time:    events/TOFBarrelHits.time
    tof_pos_x:   events/TOFBarrelHits.position.x
    # ...

  track:
    points_branch:
      - events/SiBarrelHits.position.x
      - events/SiBarrelHits.position.y
      - events/SiBarrelHits.position.z
      - events/SiBarrelHits.momentum.x
      - events/SiBarrelHits.momentum.y
      - events/SiBarrelHits.momentum.z
      - events/SiBarrelHits.pathlength
```

---

## How to Run

```bash
python analyze_script.py \
  --config   config/config.yaml \
  --output   pid_result.root \
  --filetype NCDIS            # NCDIS or single_particle_pion
```

* `--filetype` must match one of the `description` fields in `file_paths`.
* All artefacts will appear under `out/<directory_name>/`.

---

## Objects inside the output ROOT file

| Object name                              | Description                                                                    
| ---------------------------------------- | --------------------------------------- | 
| `mc_*`                                   | all MC truth branches                   |                                       
| `TOF_rec_hit_*_{btof/etof}`              | raw reconstructed TOF hits              |                                       
| `Track_trk_*`                            | all track‑point branches                |                                       
| `Track_segments_on_{btof/etof}_*`        | track points projected onto TOF (truth) |                                       
| `{btof/etof}_raw_{hit/mc}_*`             | TOF–MC matches (all particles)          |                                       
| `{btof/etof}_stable_{hit/mc}_*`          | stable particles only                   |                                       
| `{btof/etof}_reco_{hit/mc}_*`            | stable + actually reconstructed hits    |                                       
| `Matched_{track/tof/mc}_*`               | three‑way match (track↔TOF↔MC)          |                                       
| `beta_inverse_vs_p_{btof/etof}`          | β<sup>−1</sup> vs p (all)               |                                       
| `Reconstructed_Mass_{btof/etof}`         | reconstructed mass (all)                |                                       
| \`beta\_inverse\_vs\_p\_{btof/etof}\_(pi/k/p)\` | per‑species β<sup>−1</sup> vs p |
| \`Reconstructed\_Mass\_{btof/etof}\_(pi/k/p)\` | per‑species mass spectra        |

---

## Roadmap / TODO

* expose vertex cuts via YAML
* add p<sub>T</sub>‑dependent Gaussian separation plots
* full unit‑test coverage & CI


