# maptr-env 実行環境セットアップ・トラブルシューティング

このドキュメントは `maptr-env` コンテナ (image: `perception/uniad-env:v1`) 上で
`MapTR/projects/mmdet3d_plugin` を新しい PyTorch/CUDA (torch 2.8 / CUDA 12.8) で
動かすために必要だった修正をまとめたものです。

> **スコープについての注意**
> 以下の修正はすべて `MapTR/` リポジトリ内のソースファイルに対するものです。
> ただし `maptr-env` コンテナのベースイメージ (`perception/uniad-env:v1`) を
> 定義する Dockerfile はこのワークスペースには存在しません(このワークスペースにある
> `MapTR/mmdetection3d/docker/Dockerfile` はオリジナルの mmdetection3d 用テンプレートで、
> 実際に使われているイメージとは別物です)。そのため、イメージ自体の再現性はこのリポジトリの
> 変更だけでは担保できません。コンテナを作り直す場合は、後述の「コンテナ側で必要な作業」を
> 再実行してください。

## 背景

`maptr-env` コンテナは torch 2.8.0 (CUDA 12.8) を使用していますが、
`MapTR/mmdetection3d` 配下の CUDA/C++ 拡張の多くは、
`Tensor::type()` / `.data<T>()` / `THC/THC.h` など、
新しい PyTorch で削除された古い libtorch API を使って書かれていました。
また `numba.errors` のような依存ライブラリ側の API 変更、
mmdet と mmdet3d で同名バックボーンが二重登録される問題もありました。

## リポジトリ内で行った修正

### 1. GeometricKernelAttention (MapTR独自のCUDA拡張)

- [geometric_kernel_attn_cuda.cu](../MapTR/projects/mmdet3d_plugin/maptr/modules/ops/geometric_kernel_attn/src/geometric_kernel_attn_cuda.cu)
- [geometric_kernel_attn.h](../MapTR/projects/mmdet3d_plugin/maptr/modules/ops/geometric_kernel_attn/src/geometric_kernel_attn.h)

`value.type().is_cuda()` → `value.is_cuda()`、
`AT_DISPATCH_FLOATING_TYPES(value.type(), ...)` → `value.scalar_type()`、
`.data<T>()` → `.data_ptr<T>()` に置き換え。

### 2. mmdetection3d: spconv拡張をビルド対象から除外

MapTR は spconv (Sparse Convolution) を使用しないため、非互換な拡張をビルド対象から除外しました。

- [setup.py](../MapTR/mmdetection3d/setup.py): `ext_modules` から `sparse_conv_ext` を削除
  (torch 2.8 の `torch::nn::init::kaiming_uniform_` のシグネチャ変更で spconv がコンパイル不能なため)
- [mmdet3d/ops/__init__.py](../MapTR/mmdetection3d/mmdet3d/ops/__init__.py): `sparse_block` (spconv依存) の import を try/except で無効化しても壊れないように変更
- [mmdet3d/models/middle_encoders/__init__.py](../MapTR/mmdetection3d/mmdet3d/models/middle_encoders/__init__.py): `SparseEncoder`/`SparseUNet` (spconv依存) の import を try/except に変更
- [mmdet3d/models/__init__.py](../MapTR/mmdetection3d/mmdet3d/models/__init__.py): `roi_heads` (spconv依存の `PartA2BboxHead` 等を含む) の import を try/except に変更

### 3. mmdetection3d: 廃止された `THC/THC.h` の削除

以下のファイルは `THCState *state` の宣言のためだけに `THC/THC.h` を include していましたが、
実際には未使用だったため include と宣言を削除:

- `mmdet3d/ops/ball_query/src/ball_query.cpp` (`.type().is_cuda()` も `.is_cuda()` に修正)
- `mmdet3d/ops/furthest_point_sample/src/furthest_point_sample.cpp` (`.data<T>()` も `.data_ptr<T>()` に修正)
- `mmdet3d/ops/gather_points/src/gather_points.cpp`
- `mmdet3d/ops/group_points/src/group_points.cpp`
- `mmdet3d/ops/interpolate/src/interpolate.cpp`
- `mmdet3d/ops/knn/src/knn.cpp`

### 4. mmdetection3d: mmcv バージョンチェックの緩和

- [mmdet3d/__init__.py](../MapTR/mmdetection3d/mmdet3d/__init__.py): `mmcv_maximum_version` を `1.4.0` → `1.8.0` に変更
  (コンテナの mmcv==1.7.2 に対応するため)

### 5. mmdetection3d: `numba.errors` の移行先修正

- [mmdet3d/datasets/pipelines/data_augment_utils.py](../MapTR/mmdetection3d/mmdet3d/datasets/pipelines/data_augment_utils.py):
  `from numba.errors import NumbaPerformanceWarning` → `from numba.core.errors import NumbaPerformanceWarning`
  (numba 0.58 で `numba.errors` が削除されたため)

### 6. EfficientNet の二重登録回避

- [efficientnet.py](../MapTR/projects/mmdet3d_plugin/models/backbones/efficientnet.py):
  `@BACKBONES.register_module()` → `@BACKBONES.register_module(force=True)`
  (mmdet が同名の `EfficientNet` を既に登録しているため、MapTR独自実装を優先)

### 7. 学習・評価スクリプトの `--local-rank` / `--local_rank` 対応

新しい `torch.distributed.launch` / `torchrun` は `--local-rank`(ハイフン)で子プロセスに
引数を渡しますが、各スクリプトは `--local_rank`(アンダースコア)しか受け付けておらず
`unrecognized arguments: --local-rank=0` で落ちていました。両方を受け付けるよう修正:

- [tools/test.py](../MapTR/tools/test.py)
- [tools/train.py](../MapTR/tools/train.py)
- [tools/fp16/train.py](../MapTR/tools/fp16/train.py)
- [tools/maptr/test.py](../MapTR/tools/maptr/test.py)

### 8. `torch.load` の `weights_only` デフォルト変更対応

torch 2.6 以降 `torch.load` の `weights_only` デフォルトが `True` になり、
mmcv経由でnumpyスカラーを含む古い形式のチェックポイントを読み込むと
`UnpicklingError` になっていました。`torch.load` を `weights_only=False` を
デフォルトにするようモンキーパッチ(自分たちが用意した信頼できるckptなので許容):

- [tools/test.py](../MapTR/tools/test.py)
- [tools/train.py](../MapTR/tools/train.py)
- [tools/maptr/test.py](../MapTR/tools/maptr/test.py)

### 9. 評価時の `DataContainer` 展開エラー (`'DataContainer' object is not subscriptable`)

このコンテナの mmcv は `MMDistributedDataParallel` が PyTorch 2.x 向けに独自移植されており、
`train_step`/`val_step` 経由でのみ DataContainer を展開する実装になっていました。
しかし `tools/test.py` は `model(return_loss=False, ...)` のようにモデルを直接呼び出す
(= 素の `forward()`)ため、`img_metas` が `DataContainer` のまま渡り例外になっていました。

- [tools/test.py](../MapTR/tools/test.py): 評価時にモデルを `MMDistributedDataParallel` で
  ラップするのをやめ(勾配同期は不要なので単に `model.cuda()`)、
- [projects/mmdet3d_plugin/bevformer/apis/test.py](../MapTR/projects/mmdet3d_plugin/bevformer/apis/test.py):
  `mmcv.parallel.scatter_kwargs` を使って明示的に `DataContainer` を展開してから
  モデルを呼び出すように変更

### 10. CUDA arch 不一致 (`no kernel image is available for execution on the device`)

GPU が RTX 5090 (compute capability 12.0 / `sm_120`, Blackwell) である一方、
自前でビルドした CUDA 拡張(GeometricKernelAttention, bev_pool, bev_pool_v2 等)は
コンテナの環境変数 `TORCH_CUDA_ARCH_LIST=9.0;10.0` の制約で `sm_120` 向けのカーネルを
含んでおらず、実行時に `no kernel image is available for execution on the device` に
なっていました。`TORCH_CUDA_ARCH_LIST="9.0;10.0;12.0"` を指定してビルドし直すことで解決
(手順は後述の「コンテナ側で必要な作業」参照)。ソースコード自体の変更はありません。

### 11. `custom_nusc_map_converter.py`: `v1.0-mini` データセット対応

maptrv2 用のデータ変換スクリプトは `--version` に渡した値へ無条件で `-trainval`/`-test`
サフィックスを付与する作りだったため、`--version v1.0-mini` を指定すると
`v1.0-mini-trainval` という存在しないバージョン名になり失敗していました。
(`v1.0-mini` は `v1.0-trainval`/`v1.0-test` と異なり train/val が同一フォルダに同居するため。)

- [tools/maptrv2/custom_nusc_map_converter.py](../MapTR/tools/maptrv2/custom_nusc_map_converter.py):
  `--version v1.0-mini` の場合はサフィックスを付けず単独で `nuscenes_data_prep` を呼ぶよう分岐を追加

### 12. `run_visualize.sh`: 実行パスの誤り

`PYTHONPATH` がホスト側の絶対パス(`/home/.../MapTR/`)になっており、コンテナ内で実行すると
`projects` モジュールが見つからず `ModuleNotFoundError` になっていました。また
チェックポイントのパスも `MapTR/ckpts/...` と `MapTR` が重複していました。

- [run_visualize.sh](../MapTR/run_visualize.sh): `PYTHONPATH` をコンテナ内パス
  `/workspace/MapTR/` に修正し、チェックポイントパスの重複を解消

### 13. `vis_pred.py`: `MMDataParallel` のscatterが新しいtorchと非互換

`MMDataParallel(model, device_ids=[0])` 経由の `scatter_kwargs` が、内部で
`torch.nn.parallel._functions._get_stream` にint型のデバイスIDを渡していましたが、
新しいtorchではこの関数が `torch.device` オブジェクトを要求するため
`AttributeError: 'int' object has no attribute 'type'` になっていました。

- [tools/maptr/vis_pred.py](../MapTR/tools/maptr/vis_pred.py):
  - `torch.load` の `weights_only` 互換パッチを追加(項番8と同様)
  - `MMDataParallel` でのラップをやめ、`mmcv.parallel.scatter_kwargs` で
    明示的に `DataContainer` を展開してからモデルを呼び出すように変更(項番9と同様)

### 14. `vis_pred.py`: centerlineクラスの色未定義

`colors_plt = ['orange', 'b', 'g']` は3クラス(divider/ped_crossing/boundary)分しか
色を定義しておらず、centerline対応config(4クラス目)を使うと
`IndexError: list index out of range` になっていました。

- [tools/maptr/vis_pred.py](../MapTR/tools/maptr/vis_pred.py):
  `colors_plt` に4色目(centerline用、`'r'`)を追加

### 15. `tools/maptrv2/nusc_vis_pred.py` も同じ問題

`tools/maptr/vis_pred.py` の maptrv2 版である `tools/maptrv2/nusc_vis_pred.py` にも
項番8・13と同じ問題(`torch.load` の `weights_only`、`MMDataParallel` の scatter 非互換)が
あったため同様に修正(`colors_plt` は元から4色定義されており修正不要でした)。

- [tools/maptrv2/nusc_vis_pred.py](../MapTR/tools/maptrv2/nusc_vis_pred.py)

## コンテナ側で必要な作業(コンテナ再作成時に再実行が必要)

上記のソース修正を反映させるには、`maptr-env` コンテナ内で以下を実行します。

```bash
# 0. 実際のGPUに合わせて TORCH_CUDA_ARCH_LIST を指定する(RTX 5090 = sm_120 / 12.0 の例)
#    未指定/範囲外だと "no kernel image is available for execution on the device" になる
export TORCH_CUDA_ARCH_LIST="9.0;10.0;12.0"

# 1. mmdetection3d をソースから編集可能インストール(spconv以外の拡張をビルド)
cd /workspace/MapTR/mmdetection3d
pip install -e . --no-build-isolation --no-deps

# 2. 万一、旧バージョンの mmdet3d が dist-packages に実体として残っている場合は削除
#    (editable install の finder より実体パッケージが優先されてしまうため)
rm -rf /usr/local/lib/python3.11/dist-packages/mmdet3d
python -c "import mmdet3d; print(mmdet3d.__file__)"  # ソース側のパスが出ることを確認

# 3. GeometricKernelAttention (MapTR独自CUDA拡張) をビルド
cd /workspace/MapTR/projects/mmdet3d_plugin/maptr/modules/ops/geometric_kernel_attn
rm -rf build *.egg-info
python setup.py build install
```

`--no-deps` が必要な理由: `mmdet3d` の `requirements` が `numba==0.48.0` 等、
古いバージョンに固定されており、依存解決で失敗するため。

実際に使っている GPU の compute capability は以下で確認できます:

```bash
python -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
```

## 動作確認

```bash
docker exec -w /workspace/MapTR maptr-env python -c "
import torch  # 拡張(.so)を読み込む前に必ず import する
import projects.mmdet3d_plugin as plugin
import projects.mmdet3d_plugin.maptr as maptr
import projects.mmdet3d_plugin.bevformer as bevformer
print('plugin:', plugin.__file__)
print('maptr:', maptr.__file__)
print('bevformer:', bevformer.__file__)
"
```

すべて `/workspace/MapTR/...` のソースパスが表示されれば成功です。

## 既知の制限

- **spconv (Sparse Convolution) は使用不可**です。`mmdet3d.ops.spconv`,
  `SparseEncoder`, `SparseUNet`, `PartA2BboxHead` などの spconv 依存モジュールは
  import できません。MapTR (カメラ入力のみの BEV マップ推定) はこれらを使用しないため、
  通常の学習・推論には影響ありません。KITTI 系の LiDAR/spconv ベースのモデルを
  このコンテナで動かす場合は別途対応が必要です。
- コンテナのベースイメージ (`perception/uniad-env:v1`) の Dockerfile はこのワークスペース内に
  存在しないため、イメージ自体をこのリポジトリだけで再現することはできません。
