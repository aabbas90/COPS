from .config import add_panoptic_affinity_config
from .dataset_mapper import PanopticAffinityDatasetMapper
from .panoptic_seg_affinity import (
    PanopticAffinity,
    AFF_EMBED_BRANCHES_REGISTRY,
    build_aff_embed_branch,
    AMC_BRANCH_REGISTRY,
    build_amc_branch,
    PanopticAffinitySemSegHead,
    PanopticAffinityInsEmbedHead,
    PanopticAffinityAMWC
)