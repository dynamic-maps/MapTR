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

## コンテナ側で必要な作業(コンテナ再作成時に再実行が必要)

上記のソース修正を反映させるには、`maptr-env` コンテナ内で以下を実行します。

```bash
# 1. mmdetection3d をソースから編集可能インストール(spconv以外の拡張をビルド)
cd /workspace/MapTR/mmdetection3d
pip install -e . --no-build-isolation --no-deps

# 2. 万一、旧バージョンの mmdet3d が dist-packages に実体として残っている場合は削除
#    (editable install の finder より実体パッケージが優先されてしまうため)
rm -rf /usr/local/lib/python3.11/dist-packages/mmdet3d
python -c "import mmdet3d; print(mmdet3d.__file__)"  # ソース側のパスが出ることを確認

# 3. GeometricKernelAttention (MapTR独自CUDA拡張) をビルド
cd /workspace/MapTR/projects/mmdet3d_plugin/maptr/modules/ops/geometric_kernel_attn
python setup.py build install
```

`--no-deps` が必要な理由: `mmdet3d` の `requirements` が `numba==0.48.0` 等、
古いバージョンに固定されており、依存解決で失敗するため。

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
