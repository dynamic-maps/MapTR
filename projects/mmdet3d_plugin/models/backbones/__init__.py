from .vovnet import VoVNet
# RTX5090 compatibility: disabled duplicate EfficientNet registration
# from .efficientnet import EfficientNet
from .swin import SwinTransformer
__all__ = ['VoVNet']