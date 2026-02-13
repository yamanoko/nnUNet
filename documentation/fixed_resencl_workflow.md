# FixedResEncLPlanner — Usage Workflow

TaWald/nnUNet (arXiv 2410.23132v3) の ResEncL アーキテクチャを再現する固定プランナーの使い方。

## 概要

`FixedResEncLPlanner` はデータセットに依存しない固定アーキテクチャを生成するため、
事前学習→ファインチューニングにおいてアーキテクチャの不一致が起こりません。

| パラメータ | 値 |
|---|---|
| アーキテクチャ | ResidualEncoderUNet (ResEncL) — 6 stages |
| Features per stage | (32, 64, 128, 256, 320, 320) |
| Strides | \[\[1,1,1\], \[2,2,2\]×5\] |
| Kernel sizes | \[\[3,3,3\]\]×6 |
| Encoder blocks | \[1, 3, 4, 6, 6, 6\] |
| Decoder blocks | \[1, 1, 1, 1, 1\] |
| Patch size | 160×160×160 |
| Batch size | 2 |
| Spacing | 1 mm isotropic (強制) |
| 正規化 | ZScoreNormalization (全チャネル) |
| Transpose | 抑制 (常に \[0,1,2\]) |
| Configuration | 3d_fullres のみ |

---

## Step 1: 事前学習データセットの Plan & Preprocess

```bash
nnUNetv2_plan_and_preprocess -d PRETRAIN_DATASET_ID -pl FixedResEncLPlanner
```

- プランファイル `FixedResEncLPlans.json` が生成されます。
- アーキテクチャはデータセットの spacing や shape に関係なく常に同一です。

## Step 2: 事前学習 (Pretraining)

```bash
nnUNetv2_train PRETRAIN_DATASET_ID 3d_fullres all -p FixedResEncLPlans
```

- `all` を指定すると全データで学習します（cross-validation fold なし）。
- 学習完了後、`checkpoint_final.pth` が保存されます。

## Step 3: ファインチューニング先データセットの Plan & Preprocess

```bash
nnUNetv2_plan_and_preprocess -d FINETUNE_DATASET_ID -pl FixedResEncLPlanner
```

## Step 4: プランの転送 (Plans Transfer)

```bash
nnUNetv2_move_plans_between_datasets \
    -s PRETRAIN_DATASET_ID \
    -t FINETUNE_DATASET_ID \
    -sp FixedResEncLPlans \
    -tp FixedResEncLPlans
```

- 事前学習側のプラン構成（architecture セクション含む）をファインチューニング先にコピーします。
- `FixedResEncLPlanner` ではアーキテクチャが固定なので、両方のデータセットで同じプランナーを使えば
  このステップを省略しても動作します。ただし、公式ワークフローに従い実行を推奨します。

## Step 5: ファインチューニング (Finetuning)

```bash
nnUNetv2_train FINETUNE_DATASET_ID 3d_fullres FOLD \
    -p FixedResEncLPlans \
    -pretrained_weights /path/to/nnUNet_results/DatasetXXX_Name/nnUNetTrainer__FixedResEncLPlans__3d_fullres/fold_all/checkpoint_final.pth
```

- `FOLD` は `0`, `1`, `2`, `3`, `4` のいずれか、またはすべて実行して cross-validation。
- `-pretrained_weights` に Step 2 で得た checkpoint のパスを指定します。

## Step 6: 推論 (Inference)

```bash
nnUNetv2_predict \
    -i INPUT_FOLDER \
    -o OUTPUT_FOLDER \
    -d FINETUNE_DATASET_ID \
    -p FixedResEncLPlans \
    -c 3d_fullres \
    -f FOLD
```

---

## ファイル構成

```
nnunetv2/experiment_planning/experiment_planners/residual_unets/
├── residual_encoder_unet_planners.py   # 標準 ResEnc プランナー群
└── fixed_resencl_planner.py            # FixedResEncLPlanner (本ファイル)
```

## 注意事項

- **GPU メモリ**: 24 GB 以上を推奨（RTX 4090, A5000 等）。
- **プランナー名の指定**: コマンドラインでは `-pl FixedResEncLPlanner` と正確に指定してください。
- **プラン名**: 自動的に `FixedResEncLPlans` になります（変更不要）。
- **2D / 3d_lowres**: このプランナーでは `3d_fullres` のみ生成されます。
- **アーキテクチャ互換性**: TaWald/nnUNet の `get_network_from_name("ResEncL")` と全パラメータが一致します。
