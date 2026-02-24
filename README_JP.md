# Barrel TOF PID Analysis

ePIC Detector の Barrel TOF (Time Of Flight) 検出器を対象とした
PID（π/K/p 識別）性能評価スクリプトです。

```
EICrecon  →  TOF‑MC Association  →  TOF‑Track matching  →  PID calculate  →  ROOT / CSV
```

を想定しており、 本スクリプトでは、EICreconの出力ファイルに対し、各種マッチングを行い、PID評価の結果をROOT と CSV で保存します。

---

## 特長

* YAML 設定で入力ファイル／イベント数の切り替えが可能
* `uproot + awkward` を使い、EICreconの出力結果を操作
* TGraph/TH2/TH1 でヒストグラムを生成
  * β<sup>−1</sup> vs p
  * 再構成質量ヒストなど

* 出力は `out/<directory_name>/` 以下に

  * 解析結果 `< --output >.root`
  * 中間 CSV
 
csvファイルを解析することを推奨します。

---
## 解析フロー概要

```text
1) MC / Track / TOF hit 取り込み
2) TOF–MC Association  (association branch)
3) TOF–Track Matching  (θ–φ 角距離で nearest‑hit を決定)
4) PID 計算     
5) ROOT / CSV に保存
```

---
## 各モジュールの役割

| ファイル                               | 役割                                                    |  
| ------------------------------------ | ------------------------------------------------------- |
| **`analyze_script.py`**               | メイン解析ファイル      |
| **`mc_analyzer.py`**                 | MCParticles ブランチを読み込み、粒子 PDG、運動量、頂点を DataFrame 化        |
| **`tof_analyzer.py`**                | Barrel / Endcap TOF Hits を抽出。        |
| **`track_analyzer.py`**              | `_CentralTrackSegments_points` を読み込み、TOF 上でのトラックパラメータを計算 |
| **`matching_mc_and_tof.py`**         | Association ブランチを使い MC ↔ TOF hit を紐付け             |
| **`matching_tof_and_track.py`**      | θ–φ 角距離 ΔR で TOF hit ↔ Track segment をマッチ       |
| **`tof_pid_performance_manager.py`** | β, 再構成質量、PID 効率/purity/separation powerを計算しヒスト生成       |
| **`utility_function.py`**            | 座標変換や ΔR 計算などの汎用関数                                      |
| **`*_plotter.py`**                   | ROOT ヒスト / グラフ作成専用クラス                                   |

---

## 動作確認済み環境

| component | version | 備考           |
| --------- | ------- | ------------  |
| Python    | 3.10.9  | miniconda 推奨 |
| ROOT      | 6.32.02 | PyROOT 有効    |
| uproot    | 5.3.10  |               |
| awkward   | 2.6.5   |               |
| numpy     | 1.26.4  |               |
| pyyaml    | 6.0.1   |               |
| tqdm      | 4.66.4  |               |

## 使用パッケージのダウンロード

```bash
$ conda env create -f environment.yml
$ conda activate myenv
```

## ディレクトリ構成

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
|   ├── pipline_utils.py
|   ├── utility_function.py
|   ├── 〇〇_plotter.py
│   └── helper_functions.py
├── config/
│   └── config.yaml（生成タイプによって作成推奨）
├── analyze_script.py
└── out/                 
```

---

## YAML 設定ファイル例（config.yaml）

```yaml
analysis:
  directory_name: eic_pid_test
  analysis_event_type: NCDIS     
  selected_events: 10000
  verbose: true　（一部動いていない、後で確認）
  plot_verbose: true　（一部動いていない、後で確認）
  detail_plot_verbose: false
  version: "ver1_24_2"

vertex_cuts:　（未実装：現在はハードコード）
  zvtx_min: -100.0     # [mm]
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
    mc_pdg:            events/MCParticles.PDG
    mc_vertex_x:       events/MCParticles.vertex.x
    # ...
  tof:
    tof_time:          events/TOFBarrelHits.time
    tof_pos_x:         events/TOFBarrelHits.position.x
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

## 実行方法

```bash
python analyze_script.py --config ./config/minimum_config.yaml ∖
--output output.root --filetype single_particle_pion

```

* `--filetype` には YAML の `file_paths` を指定
* 生成物は `out/<directory_name>/` に保存されます

---

## 出力ファイルのTree

**は物理量

| オブジェクト名                | 内容                            |
| ---------------------- | ----------------------------- |
| `mc_**`                 | mcの全情報 |
| `TOF_rec_hit_**_{btof/etof}`         | TOFの再構成ヒットの全情報                |
| `Track_trk_**` | トラックポイントに関する全情報  |
| `Track_segments_on_{btof/etof}_**`             | TOF上でのトラックポイントの情報（truth）         |
| `{btof/etof}_raw_{hit/mc}_**`        | mcとTOFの情報を紐付けた情報       |　
| `{btof/etof}_stable_{hit/mc}_**`             | mcとTOFの情報を紐付けた情報（安定粒子）         |
| `{btof/etof}_reco_{hit/mc}_**`             | mcとTOFの情報を紐付けた情報（安定粒子かつ再構成に使われたヒットのみ）         |
| `Matched_{track/tof/mc}_**`             | trackとmcとTOFヒットのマッチング後の情報         |
| `beta_inverse_vs_p_{btof/etof}`             | PID performance (beta_inverse_vs_momentum)|
| `Reconstructed_Mass_{btof/etof}`             | PID performance (再構成質量)         |
| `beta_inverse_vs_p_{btof/etof}_(pi/k/p)`             | 各粒子毎のPID performance (beta_inverse_vs_momentum)         |
| `Reconstructed_Mass_{btof/etof}_(pi/k/p)`             | 各粒子毎のPID performance (再構成質量)         |

---
