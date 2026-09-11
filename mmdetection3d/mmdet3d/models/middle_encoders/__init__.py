# Copyright (c) OpenMMLab. All rights reserved.
from .pillar_scatter import PointPillarsScatter

__all__ = ['PointPillarsScatter']

try:
    # spconv is not built (unused by MapTR); skip if the extension is unavailable
    from .sparse_encoder import SparseEncoder
    from .sparse_unet import SparseUNet
    __all__ += ['SparseEncoder', 'SparseUNet']
except ImportError:
    pass
