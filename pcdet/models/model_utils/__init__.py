from .model_nms_utils import class_agnostic_nms
from .centernet_utils import *

# ✅ CMAE-3D 유틸리티 등록
try:
    from .radial_masking import RadialMasking
    from .hrcl_utils import (
        HRCLModule, 
        VoxelProjectionHead, 
        FrameProjectionHead,
        VoxelRelationalContrastiveLoss,
        FrameRelationalContrastiveLoss,
        HRCLLoss,
        MemoryQueue
    )
    CMAE_UTILS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ CMAE-3D utilities import failed: {e}")
    CMAE_UTILS_AVAILABLE = False