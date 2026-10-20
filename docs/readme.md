# BEVDepth + DINOv3 + Metric3Dv2：環境構築から学習・評価・可視化まで

カメラレイアウト違いの研究用ベースライン（BEVDepth の画像特徴を凍結 DINOv3、深度事前分布を凍結 Metric3Dv2 にしたもの）を、Docker イメージのビルドから学習、評価、可視化まで一通り動かすための手順です。

| 項目 | 内容 |
|---|---|
| 対象 GPU | RTX 3060 12GB（sm_86、動作確認済み） / RTX 6000 Ada 48GB（sm_89） |
| ソフトウェア | CUDA 12.1 / cuDNN 8.9 / PyTorch 2.1.2 / Python 3.11 / mmcv 2.1.0 / mmdet 3.2.0 / mmdet3d 1.4.0 / timm 1.0.30 / Lightning 2.2.5 |
| 凍結モデル | DINOv3 ViT-B/16 (`timm/vit_base_patch16_dinov3.lvd1689m`) / Metric3Dv2 ViT-S (`metric_depth_vit_small_800k.pth`) |

> 2つの GPU に対応させている仕組み
> - BEVDepth 独自の CUDA 演算（voxel pooling）は `TORCH_CUDA_ARCH_LIST="8.6;8.9"` で両方の GPU 向けにビルドします。
> - PyTorch と mmcv の公式バイナリは sm_86 向けですが、同じ世代（8.x）の sm_89 でもそのまま動きます。
> - mmcv も 8.9 向けにビルドし直したい場合は、ビルド時に `--build-arg MMCV_FROM_SOURCE=1` を付けてください（30〜60 分余計にかかります）。

---

## 0. 前提

### ホスト環境
- **Windows 11 + Docker Desktop（WSL2 バックエンド）**、または **Linux + Docker + NVIDIA Container Toolkit**
- NVIDIA ドライバ **530 以上**（CUDA 12.1 に必要）。`nvidia-smi` でバージョンを確認できます。
- ディスクの空き容量：
  - **Docker のデータ置き場に 30GB 以上**（イメージ約 20GB とビルドキャッシュ）。Windows の Docker Desktop は既定で C ドライブを使います。
  - データセット：nuScenes mini 約 4GB、trainval 約 400GB
  - 学習結果：1 回の学習あたり数 GB（ckpt 1 個が約 1.1GB）

### リポジトリ内の関連ファイル
```
BEVDepth-src/
├── .dockerignore                 data/ outputs/ ckpt などをビルドに含めない設定
├── docker/
│   ├── Dockerfile                環境定義
│   ├── requirements-pinned.txt   Python パッケージの固定バージョン
│   ├── prefetch_weights.py       凍結モデルの重みをビルド時に取得し、SHA256 を照合
│   └── entrypoint.sh             conda の有効化と、CUDA 演算が無いときの自動ビルド
├── bevdepth/exps/nuscenes/mv/
│   ├── bev_depth_dinov3.py           DINOv3 版の実験ファイル（--lr 引数あり）
│   └── bev_depth_dinov3_metric3d.py  DINOv3 + Metric3Dv2 版（--depth_prior_mode）
├── bevdepth/layers/backbones/
│   ├── dinov3_lss_fpn.py             凍結 DINOv3 + Adapter
│   └── dinov3_metric3d_lss_fpn.py    + 凍結 Metric3Dv2 による深度事前分布
└── scripts/
    ├── gen_info.py / gen_info_mini.py   nuScenes の info ファイル（.pkl）を生成
    ├── diag_heatmap.py                  1 バッチで各段の統計を出す（動作確認用）
    ├── verify_metric3d.py               Metric3D の重み、凍結状態、深度品質を検証
    ├── infer_eval.py                    推論、nuScenes 評価、画像の可視化
    └── compare_boxes.py                 GT と予測の 1 対 1 比較（クラス別・距離別・カメラ別）
```

---

## 1. イメージをビルドする

リポジトリのルートで実行します。

```bash
cd D:\BEVDepth-src            # Linux の場合は cd ~/BEVDepth-src
docker build -f docker/Dockerfile -t bevdepth-foundation:cu121 .
```

ビルド時に指定できる引数：

| 引数 | 既定値 | 説明 |
|---|---|---|
| `MAX_JOBS` | 4 | CUDA 演算をコンパイルするときの並列数。メモリが少なければ 2 にする |
| `MMCV_FROM_SOURCE` | 0 | 1 にすると mmcv を sm_86 と sm_89 向けにソースからビルドする |
| `PREFETCH_WEIGHTS` | 1 | DINOv3、Metric3Dv2、ResNet50 の重みをイメージに含める（実行時にネット接続が不要になる） |
| `METRIC3D_REF` | `eb5b6fa…` | Metric3D のコードを固定する commit。検証済みのもの |

例：`docker build -f docker/Dockerfile --build-arg MAX_JOBS=2 -t bevdepth-foundation:cu121 .`

**ビルド中に自動で確認されること**
- `mmcv._ext`（DCN などの CUDA 演算）が読み込めること
- 重みの SHA256 が公式と一致すること
  - DINOv3：`1f9ed8a2…26332`
  - Metric3Dv2：`b34b2a2b…728f6`
- BEVDepth の voxel pooling 演算が読み込めること

所要時間の目安は 20〜40 分です（`MMCV_FROM_SOURCE=1` のときは、さらに 30〜60 分）。

---

## 2. コンテナを起動する

ソースコード、データセット、結果は**ホスト側に置いてマウント**します。コンテナを作り直しても消えません。

**Windows（PowerShell）**
```powershell
docker run --gpus all -it `
  --name bevfoundation `
  --shm-size=16g `
  -p 6006:6006 `
  -v D:\BEVDepth-src:/workspace/BEVDepth `
  -v D:\datasets\nuscenes:/workspace/BEVDepth/data/nuScenes `
  bevdepth-foundation:cu121
```

**Linux**
```bash
docker run --gpus all -it \
  --name bevfoundation \
  --shm-size=16g \
  -p 6006:6006 \
  -v ~/BEVDepth-src:/workspace/BEVDepth \
  -v /data/nuscenes:/workspace/BEVDepth/data/nuScenes \
  bevdepth-foundation:cu121
```

- 起動すると conda 環境 `bevfoundation` が自動で有効になります。
- ホストのソースをマウントすると、イメージに入れておいた CUDA 演算（`.so`）が見えなくなります。その場合、**初回起動時に entrypoint が自動でビルドします**（約 1〜2 分、ログは `/tmp/bevdepth_build.log`）。
  - 3060 機と 6000 Ada 機でソースフォルダを共有する場合も、`.so` は両方の GPU 向けにビルドされるので、そのまま使えます。
- `--shm-size=16g` は DataLoader（ワーカー 4 個）に必要です。指定しないと `Bus error` で止まります。

2 回目以降の起動と、別のターミナルからの接続：
```bash
docker start bevfoundation
docker exec -it bevfoundation bash      # entrypoint を通らないので、最初に下の 1 行を実行する
conda activate bevfoundation
cd /workspace/BEVDepth
```

GPU が見えているかの確認：
```bash
python -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
# RTX 3060 → (8, 6) / RTX 6000 Ada → (8, 9)
```

---

## 3. データを準備する

nuScenes を次の構成で置きます（ホスト側 `D:\datasets\nuscenes`、コンテナ内では `data/nuScenes`）。
```
nuScenes/
├── maps/  samples/  sweeps/
├── v1.0-mini/          （mini で試す場合）
└── v1.0-trainval/      （本番の学習）
```

info ファイル（`nuscenes_infos_{train,val}.pkl`）を生成します。
```bash
cd /workspace/BEVDepth
python scripts/gen_info_mini.py   # mini：train 323 / val 81 サンプル
# python scripts/gen_info.py      # trainval：train 28130 / val 6019（v1.0-test が無いと最後でエラーになるが、train と val の pkl は生成済み）
```

> **注意**
> - mini と trainval は**同じファイル名**で上書きされます。切り替えるときは生成し直すか、事前に退避してください。
> - LiDAR 深度の GT は学習中にその場で生成するので、`depth_gt` の事前生成は不要です。

---

## 4. 動作確認（スモークテスト、約 5 分）

```bash
cd /workspace/BEVDepth
# 1 バッチの順伝播で各段の統計を表示。最後の loss_heatmap が数百程度なら正常（1 万を超えるなら異常）
PYTHONPATH=. python scripts/diag_heatmap.py dinov3

# Metric3D の重み、凍結状態、深度品質を検証
PYTHONPATH=. python scripts/verify_metric3d.py 5
#  [1] missing は mask_token の 1 個だけ（推論では使わない）、value mismatch=0
#  [2] requires_grad=True が 0 個、3 step 後に変化したテンソルが 0 個
#  [3] AbsRel が 0.10 前後（LiDAR との比較）

# 学習ループが回るか（30 step）
PYTHONPATH=. python bevdepth/exps/nuscenes/mv/bev_depth_dinov3_metric3d.py \
  --batch_size_per_device 1 --max_epochs 1 --limit_train_batches 30 \
  --limit_val_batches 0 --num_sanity_val_steps 0 \
  --default_root_dir ./outputs/smoke
```

---

## 5. 学習する

### 推奨設定

| | RTX 3060 12GB | RTX 6000 Ada 48GB |
|---|---|---|
| `--batch_size_per_device` | 1 | 8（目安。メモリに応じて調整） |
| `--lr` | 1e-4 | 2e-4（目安） |
| 速度の実測値 / 目安 | 約 0.6〜0.7 it/s（mini 1 epoch 約 8 分） | 未計測 |
| ピークメモリ | 約 4.5GB | — |

- `--lr` を指定しないと、元の BEVDepth の式 `2e-4/64 × バッチサイズ` が使われます。**バッチ 1 では 3e-6 になり、ほとんど学習が進みません**。必ず指定してください。
- 学習率は 24 epoch 中の 19 epoch 目と 23 epoch 目に 1/10 に下がります。この位置は `--max_epochs` に合わせて自動で決まります。
- 6000 Ada の列の値は**まだ試していない目安**です。最初に `--limit_train_batches 30` で速度とメモリを確認してから決めてください。

### 実行（コンテナ内、ターミナルを閉じても止まらないよう nohup を使う）
```bash
cd /workspace/BEVDepth
EXP=dinov3_metric3d_concat_mini_e24
nohup env PYTHONPATH=. python -u bevdepth/exps/nuscenes/mv/bev_depth_dinov3_metric3d.py \
  --batch_size_per_device 1 --lr 1e-4 --max_epochs 24 \
  --depth_prior_mode concat \
  --limit_val_batches 0 --num_sanity_val_steps 0 \
  --default_root_dir ./outputs/$EXP \
  > train_$EXP.log 2>&1 &
tail -f train_$EXP.log
```

ホスト側から直接起動する場合（コンテナの中に入らずに済みます）：
```bash
docker exec -d bevfoundation bash -lc 'source /opt/conda/etc/profile.d/conda.sh && conda activate bevfoundation && cd /workspace/BEVDepth && PYTHONPATH=. python -u bevdepth/exps/nuscenes/mv/bev_depth_dinov3_metric3d.py --batch_size_per_device 1 --lr 1e-4 --max_epochs 24 --depth_prior_mode concat --limit_val_batches 0 --num_sanity_val_steps 0 --default_root_dir ./outputs/dinov3_metric3d_concat_mini_e24 > train_dinov3_metric3d_concat_mini_e24.log 2>&1'
```

### 実験の種類

| 実験ファイル | 内容 |
|---|---|
| `bev_depth_lss_r50_256x704_128x128_24e_2key.py` など | 元の BEVDepth（ResNet50） |
| `bev_depth_dinov3.py` | 画像特徴を凍結 DINOv3 に置き換え |
| `bev_depth_dinov3_metric3d.py --depth_prior_mode concat` | + Metric3D の深度分布を DepthNet に入力し、DepthNet が補正を学習 |
| `bev_depth_dinov3_metric3d.py --depth_prior_mode replace` | + Metric3D の深度分布をそのまま LSS の深度として使う（学習した深度推定を外す切り分け実験用） |

### 途中経過を見る（TensorBoard）
```bash
# コンテナ内
tensorboard --logdir outputs --bind_all --port 6006
```
ブラウザで http://localhost:6006 を開きます。見るべき指標は `train/loss_heatmap`、`train/loss_bbox`、`depth_loss`、`train/total_loss` です。

### 途中の重みを残す
Lightning は `lightning_logs/version_*/checkpoints/` に**最新の 1 個だけ**を残し、古いものは消します。途中の重みで評価したいときは、先にコピーしておきます。
```bash
mkdir -p outputs/$EXP/snapshots
cp outputs/$EXP/lightning_logs/version_0/checkpoints/*.ckpt outputs/$EXP/snapshots/
```

---

## 6. 評価と可視化

> Lightning 2.x では `-e`（`trainer.test`）を使っても評価結果が出ません（`test_epoch_end` が呼ばれないため）。
> 代わりに `scripts/infer_eval.py` を使ってください。

### 6-1. 推論、nuScenes 評価、画像の可視化
```bash
cd /workspace/BEVDepth
EXP=dinov3_metric3d_concat_mini_e24
CKPT=outputs/$EXP/snapshots/epoch=1-step=646.ckpt     # 評価したい ckpt
OUT=outputs/$EXP/eval_epoch1

PYTHONPATH=. python scripts/infer_eval.py \
  --exp dinov3_metric3d --ckpt "$CKPT" --out $OUT \
  --num_vis 10 --score_thr 0.3 --version v1.0-mini
```

| オプション | 説明 |
|---|---|
| `--exp` | `dinov3_metric3d` / `dinov3` / `resnet` のいずれか（学習に使った実験ファイルに合わせる） |
| `--version` | `v1.0-mini` または `v1.0-trainval` |
| `--num_vis` | 何サンプル分 PNG を出すか |
| `--score_thr` | 可視化に表示する予測のスコアの下限（mAP の計算は全予測で行う） |
| `--depth_prior_mode` | replace モードで学習した ckpt を評価するときは `replace` を指定する |

ターミナルに mAP、NDS、誤差指標（mATE、mASE、mAOE、mAVE、mAAE）、クラス別 AP が表示されます。

### 6-2. GT と予測の 1 対 1 比較
```bash
PYTHONPATH=. python scripts/compare_boxes.py \
  --pred $OUT/raw_results.pkl --out $OUT/compare \
  --score_thr 0.3 --dist_thr 2.0 --num_vis 10
```
- 同じクラスの中で、スコアの高い予測から順に、BEV 上の中心距離が `--dist_thr`（m）以内の GT と対応づけます。
- 結果はクラス別、距離別（0-20 / 20-30 / 30-40 / 40-52m）、**カメラ別**（GT の中心がどのカメラに写っているか）に集計されます。

### 6-3. 出力ファイル
```
outputs/<EXP>/
├── lightning_logs/version_0/
│   ├── checkpoints/*.ckpt        最新の重み
│   ├── events.out.tfevents.*     TensorBoard 用ログ
│   └── hparams.yaml              学習時のハイパーパラメータ
├── snapshots/*.ckpt              手動で退避した途中の重み
└── eval_<tag>/
    ├── raw_results.pkl           予測ボックスの生データ（compare_boxes.py の入力）
    ├── results_nusc.json         nuScenes 提出形式（グローバル座標）
    ├── metrics_summary.json      mAP、NDS、クラス別 AP、TP 誤差
    ├── metrics_details.json      クラス×距離しきい値ごとの詳細
    ├── vis/NNN.png               ① カメラ画像と BEV の可視化
    └── compare/
        ├── summary.txt           クラス別・距離別・カメラ別の集計
        ├── matches.csv           GT 1 件 = 1 行（対応した予測と誤差。未検出は空欄）
        ├── false_pos.csv         誤検出の一覧
        └── bev_match/NNN.png     ② GT と予測の対応図
```

**① `vis/NNN.png` の見方**

| 位置 | 内容 |
|---|---|
| 上段・中段 | 6 カメラ画像。緑の実線が GT、クラス色の破線が予測（score ≥ 閾値） |
| 右上 | BEV（自車中心で ±51.2m、x が前方、y が左）。灰色の点は LiDAR |
| 下段 | CAM_FRONT の深度。左から Metric3Dv2、LSS の深度期待値（DepthNet）、LiDAR 正解（いずれも 0〜60m） |

**② `compare/bev_match/NNN.png` の見方**

| 表示 | 意味 |
|---|---|
| 緑の塗りつぶし | 検出できた GT（TP） |
| 赤の塗りつぶし | 見逃した GT（FN） |
| 青の破線 | GT と対応した予測 |
| 橙の破線 | 誤検出（FP） |
| 黒い線 | GT と予測の中心のずれ |
| 点線の円 | 自車から 20 / 30 / 40m |
| ▶ | 自車 |

ファイル番号は、`vis/` と `bev_match/` のどちらも val のサンプル番号です。

---

## 7. トラブルシューティング

| 症状 | 原因と対処 |
|---|---|
| `loss_heatmap` が 1 万を超えたまま下がらない | heatmap の bias が初期化されていない。`bev_depth_head.py` の `__init__` に bias -2.19 の設定があるか確認する |
| `Bus error` / DataLoader が止まる | `--shm-size=16g` を付け忘れている |
| `ImportError: voxel_pooling_train_ext` | 拡張が未ビルド。`python setup.py develop` を実行し、ログを確認する |
| ビルド中に `No space left on device` | Docker のデータ置き場（Windows では C ドライブ）の空きが足りない。`docker system df` で確認し、不要なイメージを削除するか、Docker Desktop の設定で置き場を別ドライブに移す |
| `entrypoint.sh: not found` / `\r` に関するエラー | Windows の改行コード（CRLF）が原因。Dockerfile 内で変換しているが、手でコピーした場合は LF に変換する |
| 学習中に OOM | `--batch_size_per_device` を下げる。凍結モデルはカメラ 1 台ずつ処理している（`camera_chunk_size=1`） |
| 実行時に重みをダウンロードしようとする | `PREFETCH_WEIGHTS=0` でビルドしている。ネット接続があれば自動で取得する |

---

## 付録：検証済みの内容（2026-10-20、RTX 3060）

| 項目 | 結果 |
|---|---|
| DINOv3 の重み | HuggingFace 公式と SHA256 が一致。strict で読み込み、凍結されている |
| Metric3Dv2 の重み | HuggingFace 公式と SHA256 が一致。361/362 テンソルを読み込み（未ロードは推論で使わない `mask_token` のみ）。3 step 学習後も全テンソルが不変 |
| Metric3Dv2 の深度品質（val、LiDAR と比較、1〜60m） | 元画像 1600×900：AbsRel 0.090 / モデル内 256×704：AbsRel 0.103、δ<1.25 0.93 |
| 修正済みのバグ | ① heatmap の bias が初期化されていない（初期 loss 17,000 → 440） ② timm の `features_only` で DINOv3 の最終 LayerNorm が外される ③ Metric3D の焦点距離が flip 拡張で負になる＋前処理のずれ |
| RTX 6000 Ada | イメージは sm_89 向けにも作っているが、**実機ではまだ試していない** |
