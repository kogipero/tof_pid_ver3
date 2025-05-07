# Barrel TOF PID Analysis

ePIC Detector の Barrel TOF (Time Of Flight) 検出器を対象とした
**PID（π/K/p 識別）性能評価スクリプト**です。

MC(dd4hep) → 再構成シミュレーション（EICrecon） → 各種マッチング → PID 評価
を想定しており、 再構成シミュレーションの出力ファイルに対し、各種マッチングを行い、PID評価の結果をROOT と CSV で保存します。

---

## 特長

* **YAML 設定** で入力ファイル／TTreeの名前／カット条件の切り替えが可能
* `uproot + awkward` を使い、EICreconの出力結果を操作
* TGraph/TH2/TH1 でヒストグラムを生成

  * β<sup>−1</sup> vs p
  * 再構成質量ヒスト
  * PID 効率・純度・分離能 vs p<sub>T</sub>

* 出力は `out/<directory_name>/` 以下に

  * 解析結果 `< --output >.root`
  * 中間 CSV（TOF ↔ Track マッチング結果など）※必要に応じて削除してください

---

## 動作確認済み環境

| component | version | 備考           |
| --------- | ------- | ------------ |
| Python    | 3.10    | miniconda 推奨 |
| ROOT      | 6.30/04 | PyROOT 有効    |
| uproot    | 5.x     | 要確認         |
| awkward   | 2.x     | 要確認         |
| numpy     | 1.26+   | 要確認         |
| pyyaml    | 6.x     | 要確認         |
| tqdm      | 4.x     | 要確認         |


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
│   └── helper_functions.py
├── config/
│   └── config.yaml（生成タイプによって作成推奨）
├── analyze_script.py
└── out/                 # 出力先 (git ignore)
```

---

## YAML 設定ファイル例（config.yaml）

```yaml
analysis:
  directory_name: eic_pid_test
  analysis_event_type: NCDIS            # run_analysis.py の --filetype で上書き可
  selected_events: 10000
  verbose: true
  plot_verbose: true
  detail_plot_verbose: false
  version: "ver1_24_2"

vertex_cuts:
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
python analyze_script.py \
  --config   config/config.yaml \
  --output   pid_result.root \
  --filetype NCDIS
```

* `--filetype` には YAML の `file_paths[*].description` を指定
* 生成物は `out/<directory_name>/` に保存されます

---

## 出力 ROOT ファイルの例

| オブジェクト名                | 内容                            |
| ---------------------- | ----------------------------- |
| `beta_inv_vs_p_btof`   | β<sup>−1</sup> vs p (BTOF 全体) |
| `mass_pi_btof`         | π 再構成質量 (BTOF)                |
| `beta_inv_vs_p_k_etof` | K β<sup>−1</sup> vs p (ETOF)  |
| `trk_path`             | トラック区間 pathlength ヒスト         |
| `c_purity_btof`        | π/K/p 純度 vs p (TCanvas)       |　（未実装）

---

## 今後のアップデート予定

* 設定ファイルの変更
* separation vs pt分布の実装
