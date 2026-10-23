# RTX 6000 Ada で学習・推論を動かすまでの手順（ステップバイステップ）

対象：RTX 6000 Ada（48GB、sm_89）を積んだマシンで、`docker/Dockerfile` を使って BEVDepth + DINOv3 + Metric3Dv2 の学習・推論・評価・可視化を動かす人。

> **現時点の検証状況（2026-10-23）**
> - この Dockerfile は **まだ一度もビルドしていません**（開発機の C ドライブの空きが足りないため）。
> - 確認済みなのは次の点です。
>   - ダウンロード元（ベースイメージ、torch の wheel、Miniforge、重み3つ）がすべて取得できる
>   - 重みの事前ダウンロード用スクリプトが、3つとも公式と同じ SHA256 で取得できる
>   - 固定バージョンの組み合わせで、RTX 3060 上の学習・推論が動く
> - **RTX 6000 Ada の実機では未確認**です。ステップ 4 と 7 の確認結果を必ず記録してください。

全体の流れ：

| ステップ | 内容 | 目安時間 |
|---|---|---|
| 1 | ホスト環境の確認 | 10 分 |
| 2 | ソースコードを 6000 Ada 機に持ってくる | 10 分 |
| 3 | データセットを置く | データ量次第 |
| 4 | Docker イメージをビルドして確認する | 30〜60 分 |
| 5 | コンテナを起動する | 5 分 |
| 6 | info ファイルを生成する | mini 1 分 / trainval 30 分程度 |
| 7 | 動作確認と速度計測 | 15 分 |
| 8 | 学習する | 下の表を参照 |
| 9 | 推論・評価・可視化 | mini 数分 / trainval 2 時間程度 |
| 10 | 結果を持ち帰る | — |

---

## ステップ 1：ホスト環境を確認する

### 1-1. NVIDIA ドライバ（530 以上が必要）
```bash
nvidia-smi
```
- 右上の `Driver Version` が **530 以上**であることを確認します（イメージが CUDA 12.1 を使うため）。
- `NVIDIA RTX 6000 Ada Generation` と `48GB` 前後のメモリが表示されることを確認します。

### 1-2. Docker と、コンテナから GPU を使う仕組み
**Linux の場合**：Docker Engine と NVIDIA Container Toolkit が必要です。
```bash
docker --version
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
```
2つ目のコマンドで、コンテナの中から GPU が見えれば OK です。
- `could not select device driver "" with capabilities: [[gpu]]` と出た場合は、NVIDIA Container Toolkit が入っていません。インストールしてから `sudo systemctl restart docker` を実行してください。

**Windows の場合**：Docker Desktop（WSL2 バックエンド）を使います。上の2つのコマンドを PowerShell で実行して確認します。

### 1-3. ディスクの空き容量
| 用途 | 必要量 |
|---|---|
| Docker のデータ置き場（イメージとビルドキャッシュ） | **30GB 以上** |
| nuScenes mini / trainval | 約 4GB / 約 400GB |
| 学習結果（ckpt 1 個が約 1.1GB） | 1 回の学習あたり数 GB〜数十 GB |

確認方法：Linux は `df -h /var/lib/docker`、Windows は Docker Desktop の Settings → Resources → Disk image location で場所を確認します。

---

## ステップ 2：ソースコードを 6000 Ada 機に持ってくる

`D:\BEVDepth-src` は git 管理されていないので、フォルダごと転送します。**`outputs/`、`data/`、`build/` などの大きなものは除いて**送ります。

**開発機（Windows、Git Bash）で圧縮する**
```bash
cd /d
tar --exclude='BEVDepth-src/outputs' --exclude='BEVDepth-src/data' \
    --exclude='BEVDepth-src/build' --exclude='BEVDepth-src/work_dirs' \
    --exclude='BEVDepth-src/workspace' --exclude='*.ckpt' --exclude='*.log' \
    --exclude='__pycache__' \
    -czf BEVDepth-src.tar.gz BEVDepth-src
```

**6000 Ada 機で展開する**（以下、Linux で `~/BEVDepth-src` に置く前提で書きます）
```bash
cd ~ && tar -xzf BEVDepth-src.tar.gz
ls ~/BEVDepth-src/docker
# Dockerfile  entrypoint.sh  prefetch_weights.py  requirements-pinned.txt  patches/  requirements.txt
```

今回の研究で追加・修正したファイルが入っていることを確認します。
```bash
cd ~/BEVDepth-src
ls bevdepth/layers/backbones/dinov3_lss_fpn.py \
   bevdepth/layers/backbones/dinov3_metric3d_lss_fpn.py \
   bevdepth/exps/nuscenes/mv/bev_depth_dinov3.py \
   bevdepth/exps/nuscenes/mv/bev_depth_dinov3_metric3d.py \
   scripts/infer_eval.py scripts/compare_boxes.py \
   scripts/plot_loss.py scripts/fixed_set_loss.py
grep -n "task_head.heatmap\[-1\].bias" bevdepth/layers/heads/bev_depth_head.py   # heatmap bias の修正が入っているか
```

> 開発機でビルドした CUDA 演算の `.so` ファイル（`bevdepth/ops/*/*.so`）が一緒に来ても問題ありません（sm_86 / sm_89 の両方向けにビルド済み）。
> 来なかった場合は、ステップ 5 で自動的にビルドされます。

---

## ステップ 3：データセットを置く

nuScenes を次の構成で置きます（例：`/data/nuscenes`）。
```
/data/nuscenes/
├── maps/
├── samples/
├── sweeps/
├── v1.0-mini/        （mini で試す場合）
└── v1.0-trainval/    （本番の学習）
```
開発機の `D:\datasets\nuscenes` を持ってくるか、nuScenes の公式サイトからダウンロードして展開します。

---

## ステップ 4：Docker イメージをビルドして確認する

### 4-1. ビルド
```bash
cd ~/BEVDepth-src
docker build -f docker/Dockerfile \
  --build-arg MAX_JOBS=8 \
  -t bevdepth-foundation:cu121 .
```
- `MAX_JOBS`：CUDA 演算をコンパイルするときの並列数です。CPU コア数とメモリに合わせて 4〜16 にします。
- **mmcv も sm_89 向けにビルドし直したい場合**は、`--build-arg MMCV_FROM_SOURCE=1` を足します（30〜60 分余計にかかります）。
  - 既定の公式 wheel（sm_86 向け）も、同じ世代の sm_89 でそのまま動きます。
  - ステップ 4-2 で `no kernel image is available` が出たときだけ、この方法でビルドし直してください。

ビルド中に次の確認が自動で行われます。どれかが失敗すると、ビルドがエラーで止まります。
- `torch 2.1.2+cu121 mmcv 2.1.0 mmdet 3.2.0 mmdet3d 1.4.0` の表示
- `[OK] DINOv3 ... sha256=1f9ed8a2...`、`[OK] Metric3Dv2 ... sha256=b34b2a2b...`、`[OK] ResNet50`
- `BEVDepth CUDA ops OK`

### 4-2. ビルド後の確認（GPU を使って実行）
```bash
docker run --rm --gpus all bevdepth-foundation:cu121 python -c "
import torch, mmcv, mmcv._ext
print('torch', torch.__version__, 'cuda', torch.version.cuda)
print('device', torch.cuda.get_device_name(0), 'capability', torch.cuda.get_device_capability(0))
print('arch list', torch.cuda.get_arch_list())
x = torch.randn(1, 3, 64, 64, device='cuda')
from mmcv.ops import DeformConv2dPack
m = DeformConv2dPack(3, 8, 3, padding=1).cuda()
print('mmcv DCN on GPU OK', m(x).shape)
from bevdepth.ops.voxel_pooling_train import voxel_pooling_train
print('voxel pooling import OK')
"
```
期待される出力：
- `capability (8, 9)`
- `mmcv DCN on GPU OK torch.Size([1, 8, 64, 64])`
- `voxel pooling import OK`

---

## ステップ 5：コンテナを起動する

```bash
docker run --gpus all -it \
  --name bevfoundation \
  --shm-size=32g \
  -p 6006:6006 \
  -v ~/BEVDepth-src:/workspace/BEVDepth \
  -v /data/nuscenes:/workspace/BEVDepth/data/nuScenes \
  bevdepth-foundation:cu121
```
- **`--shm-size=32g`**：DataLoader（ワーカー 4 個）が共有メモリを使います。バッチを大きくするので、3060 機（16g）より多めにします。
- ソースはホストからマウントします。コンテナを消しても、コードと結果はホスト側に残ります。
- 起動時に `[entrypoint] building BEVDepth CUDA ops` と出た場合は、CUDA 演算を自動でビルドしています（1〜2 分）。ログは `/tmp/bevdepth_build.log` にあります。

2 回目以降の起動と、別のターミナルからの接続：
```bash
docker start bevfoundation
docker exec -it bevfoundation bash
conda activate bevfoundation          # docker exec で入ったときは毎回必要
cd /workspace/BEVDepth
```

---

## ステップ 6：info ファイルを生成する（コンテナ内）

```bash
cd /workspace/BEVDepth
python scripts/gen_info_mini.py      # mini：train 323 / val 81
# python scripts/gen_info.py         # trainval：train 28130 / val 6019
ls -la data/nuScenes/nuscenes_infos_{train,val}.pkl
```
- mini と trainval は**同じファイル名で上書き**されます。切り替えるときは生成し直してください。
- `gen_info.py` は最後に `v1.0-test` を読みに行きます。test を置いていないとエラーになりますが、train と val の pkl はその時点で生成済みなので問題ありません。

---

## ステップ 7：動作確認と速度計測

### 7-1. 1 バッチの順伝播（約 2 分）
```bash
PYTHONPATH=. python scripts/diag_heatmap.py dinov3 2>&1 | grep -E "^(img|dinov3|adapter|bev|task|loss)"
```
最後の行の `loss_heatmap` が **数百程度**なら正常です（1 万を超えるなら異常）。

### 7-2. Metric3D の重み・凍結状態・深度品質の検証（約 5 分）
```bash
PYTHONPATH=. python scripts/verify_metric3d.py 5 2>&1 | grep -E "^(=|\[|  )" | grep -v PretrainedInit
```
次の結果になれば OK です。
- `[1] missing` は `mask_token` の 1 個だけ、`value mismatch=0`
- `[2] requires_grad=True` が 0 個、3 step 後に変化したテンソルが 0 個
- `[3] AbsRel` が 0.10 前後

### 7-3. 速度計測（30 step × 2 通り、約 5 分）
```bash
# (a) 3060 と同じ条件（バッチ 1）
PYTHONPATH=. python bevdepth/exps/nuscenes/mv/bev_depth_dinov3_metric3d.py \
  --batch_size_per_device 1 --lr 1e-4 --depth_prior_mode lift \
  --max_epochs 1 --limit_train_batches 30 --limit_val_batches 0 \
  --num_sanity_val_steps 0 --default_root_dir ./outputs/bench_ada_bs1 \
  2>&1 | tee bench_ada_bs1.log | grep -ao "30/30 \[[^]]*\]" | tail -1

# (b) バッチ 8
PYTHONPATH=. python bevdepth/exps/nuscenes/mv/bev_depth_dinov3_metric3d.py \
  --batch_size_per_device 8 --lr 2e-4 --depth_prior_mode lift \
  --max_epochs 1 --limit_train_batches 30 --limit_val_batches 0 \
  --num_sanity_val_steps 0 --default_root_dir ./outputs/bench_ada_bs8 \
  2>&1 | tee bench_ada_bs8.log | grep -ao "30/30 \[[^]]*\]" | tail -1

# 計測中に別のターミナルで GPU メモリを見る
watch -n 2 nvidia-smi
```

**記録する項目**（3060 との比較用）：

| 条件 | it/s | サンプル/秒（= it/s × バッチ） | GPU メモリのピーク |
|---|---|---|---|
| RTX 3060、バッチ 1（参考） | 0.71 | 0.71 | 約 4.5GB |
| 6000 Ada、バッチ 1 | | | |
| 6000 Ada、バッチ 8 | | | |

- サンプル/秒 からエポックあたりの時間を計算できます：mini は 323 ÷ サンプル/秒、trainval は 28,130 ÷ サンプル/秒（秒）。
- バッチ 8 でメモリが余っていれば 12、16 も試せます。OOM になったら下げてください。
- `nvidia-smi` で **GPU 使用率が低い（< 70%）**ままなら、CPU 側のデータ読み込みが追いついていません（DataLoader のワーカー数は `bevdepth/exps/nuscenes/base_exp.py` の `num_workers=4` で固定です）。

---

## ステップ 8：学習する

### 8-1. 実行
ターミナルを閉じても止まらないよう、**ホスト側から `docker exec -d`** で起動します。
```bash
EXP=dinov3_metric3d_lift_trainval_e24_bs8
docker exec -d bevfoundation bash -lc "source /opt/conda/etc/profile.d/conda.sh && conda activate bevfoundation && cd /workspace/BEVDepth && \
  PYTHONPATH=. python -u bevdepth/exps/nuscenes/mv/bev_depth_dinov3_metric3d.py \
  --batch_size_per_device 8 --lr 2e-4 --max_epochs 24 \
  --depth_prior_mode lift \
  --limit_val_batches 0 --num_sanity_val_steps 0 \
  --default_root_dir ./outputs/$EXP > train_$EXP.log 2>&1"
tail -f ~/BEVDepth-src/train_$EXP.log
```

| 引数 | 値 | 説明 |
|---|---|---|
| `--depth_prior_mode` | `lift` / `concat` / `replace` | lift：DepthNet なし、concat：DepthNet で補正、replace：Metric3D の深度をそのまま使い DepthNet は context のみ |
| `--batch_size_per_device` | 8（目安） | ステップ 7-3 の結果で決める |
| `--lr` | バッチ 1 なら 1e-4、バッチ 8 なら 2e-4（目安） | **必ず指定する**（指定しないとバッチ 1 で 3e-6 になり、ほとんど学習しない） |
| `--max_epochs` | 24 | 学習率は 19/24 と 23/24 の位置で 1/10 に下がる（自動） |
| `--depth_scale` / `--depth_sigma_k` / `--depth_sigma_min` | 0.958 / 0.18 / 0.5 | ガウス深度分布の μ と σ。mini の学習データで較正した値 |

### 8-2. 途中経過を見る（TensorBoard）
```bash
docker exec -d bevfoundation bash -lc "source /opt/conda/etc/profile.d/conda.sh && conda activate bevfoundation && \
  tensorboard --logdir /workspace/BEVDepth/outputs --bind_all --port 6006"
```
ブラウザで `http://<6000 Ada 機の IP>:6006` を開きます。見る指標は `train/loss_heatmap`、`train/loss_bbox`、`detection_loss`、`depth_loss_monitor`（lift）または `depth_loss`（concat）です。

### 8-3. 途中の重みを残す
Lightning は最新の ckpt を 1 個しか残しません。途中の重みで評価したいときは、先にコピーしておきます。
```bash
mkdir -p outputs/$EXP/snapshots
cp outputs/$EXP/lightning_logs/version_0/checkpoints/*.ckpt outputs/$EXP/snapshots/
```

### 8-4. 止まってしまったときの再開
```bash
# 8-1 のコマンドに、次の引数を足して実行する
--ckpt_path outputs/$EXP/lightning_logs/version_0/checkpoints/<最新>.ckpt
```

### 学習前に知っておくべきこと（未修正の既知の問題）
| 問題 | 影響 | 対処 |
|---|---|---|
| 学習用 DataLoader が `shuffle=False`（`base_exp.py` の `train_dataloader`） | GPU 1 枚ではシーン順に学習され、学習が偏り、loss 曲線も読みにくくなる | 修正は未承認。`--devices 2` 以上（DDP）なら Lightning が自動でシャッフルする |
| ガウス深度分布の μ と σ は mini で較正した値 | trainval では最適でない可能性がある | trainval では `scripts/calib_metric3d_sigma.py` で較正し直し、`--depth_scale` などで指定する |
| Metric3D を毎 step 推論している | 1 step の約 6 割を占める | 事前計算の実装は検討中 |

---

## ステップ 9：推論・評価・可視化

> Lightning 2.x では `-e`（`trainer.test`）を使っても評価結果が出ません。以下のスクリプトを使ってください。

```bash
cd /workspace/BEVDepth
EXP=dinov3_metric3d_lift_trainval_e24_bs8
CKPT=$(ls outputs/$EXP/lightning_logs/version_0/checkpoints/*.ckpt | tail -1)
OUT=outputs/$EXP/eval_final
VER=v1.0-trainval          # mini で学習した場合は v1.0-mini

# 9-1. 推論、nuScenes 評価（mAP / NDS）、画像の可視化
PYTHONPATH=. python scripts/infer_eval.py --exp dinov3_metric3d \
  --depth_prior_mode lift --ckpt "$CKPT" --out $OUT \
  --version $VER --num_vis 20 --score_thr 0.3

# 9-2. GT と予測の 1 対 1 比較（クラス別・距離別・カメラ別）
PYTHONPATH=. python scripts/compare_boxes.py \
  --pred $OUT/raw_results.pkl --out $OUT/compare \
  --score_thr 0.3 --dist_thr 2.0 --num_vis 20

# 9-3. 学習中の loss の推移と、減少の統計的な判定
python scripts/plot_loss.py \
  --logdir outputs/$EXP/lightning_logs/version_0 \
  --out outputs/$EXP/loss_curve.png --title "$EXP"

# 9-4. 同じ val サンプルで、学習前と学習後の loss を比較する
PYTHONPATH=. python scripts/fixed_set_loss.py --exp dinov3_metric3d \
  --depth_prior_mode lift --ckpts init "$CKPT" \
  --labels "init (untrained)" "final" --out outputs/$EXP/fixed_set_loss.png
```
- `--depth_prior_mode` は、学習に使ったモードと同じものを指定してください。
- trainval の val（6,019 サンプル）では、9-1 に 1〜2 時間かかる見込みです。9-4 はその倍かかるので、必要なときだけ実行してください。

出力ファイル：
```
outputs/<EXP>/
├── loss_curve.png             9-3：学習中の loss 推移と統計判定
├── fixed_set_loss.png / .csv  9-4：同じサンプルでの学習前後の比較
└── eval_final/
    ├── metrics_summary.json   mAP、NDS、クラス別 AP
    ├── vis/NNN.png            6 カメラ画像と BEV の可視化、深度 3 種類（Metric3D / Lift に入った深度 / LiDAR）
    └── compare/
        ├── summary.txt        クラス別・距離別・カメラ別の集計
        ├── matches.csv / false_pos.csv
        └── bev_match/NNN.png  BEV 上の対応図（TP は緑、FN は赤、FP は橙）
```

---

## ステップ 10：結果を持ち帰る

ckpt を除いて圧縮すると軽くなります（ckpt は 1 個約 1.1GB）。
```bash
cd ~/BEVDepth-src
tar --exclude='*.ckpt' -czf results_$EXP.tar.gz outputs/$EXP train_$EXP.log
```

---

## トラブルシューティング（6000 Ada 特有のものを中心に）

| 症状 | 原因 | 対処 |
|---|---|---|
| `CUDA driver version is insufficient for CUDA runtime version` | ホストのドライバが 530 未満 | ドライバを 530 以上に更新する |
| `no kernel image is available for execution on the device` | sm_89 で動くバイナリがない（想定外） | `--build-arg MMCV_FROM_SOURCE=1` でビルドし直す。BEVDepth の演算なら、コンテナ内で `TORCH_CUDA_ARCH_LIST="8.6;8.9" python setup.py develop` |
| `could not select device driver ... [[gpu]]` | NVIDIA Container Toolkit が入っていない | インストールして Docker を再起動する |
| `Bus error` / DataLoader が止まる | 共有メモリ不足 | `--shm-size` を増やす（32g → 64g） |
| CUDA out of memory | バッチが大きすぎる | `--batch_size_per_device` を下げる |
| GPU 使用率が低く、速くならない | CPU 側のデータ読み込みが遅い | `base_exp.py` の `num_workers` を増やす（CPU コア数まで） |
| `ImportError: voxel_pooling_train_ext` | CUDA 演算が未ビルド | コンテナ内で `python setup.py develop` を実行し、ログを確認する |
| `entrypoint.sh: not found` / `\r` に関するエラー | 改行コードが CRLF になっている | Dockerfile 内で変換しているが、手で直す場合は `sed -i 's/\r$//' docker/entrypoint.sh` |
| 実行時に重みのダウンロードが始まる | `PREFETCH_WEIGHTS=0` でビルドした、または環境変数 `HF_HOME` / `TORCH_HOME` を上書きした | ネットに接続していれば自動で取得する。オフラインならビルドし直す |
| `loss_heatmap` が 1 万を超えたまま | heatmap bias の修正が入っていない古いソース | ステップ 2 の `grep` で修正が入っているか確認する |

---

## 開発機（RTX 3060）との条件の違い（比較するときの注意）

| 項目 | RTX 3060（開発機） | RTX 6000 Ada |
|---|---|---|
| バッチ / lr | 1 / 1e-4 | 8 / 2e-4（目安） |
| 1 epoch の step 数（mini） | 323 | 40（323 ÷ 8） |
| BatchNorm の統計 | バッチ 1 で計算 | バッチ 8 で計算（より安定） |
| `--shm-size` | 16g | 32g |

バッチと lr を変えると、3060 での結果とは学習条件が変わります。2 台の結果を比べる場合は、どちらかに条件をそろえてください。
