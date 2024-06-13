from .base.base import BaseProvider
from .base.base_embedding import EmbeddingProvider
from .base.base_llm import LLMProvider

from .llm.openai import OpenAIProvider
from .llm.claude import ClaudeProvider
from .llm.restful_claude import RestfulClaudeProvider

from .object_detect.gd_provider import GdProvider

from .video.video_easyocr_extract import VideoEasyOCRExtractProvider
from .video.video_record import VideoRecordProvider
from .video.video_frame_extract import VideoFrameExtractProvider
from .video.video_clip import VideoClipProvider

from .module.information_gathering import RDR2InformationGatheringProvider, InformationGatheringProvider, StardewInformationGatheringProvider
from .module.self_reflection import RDR2SelfReflectionProvider, SelfReflectionProvider, StardewSelfReflectionProvider
from .module.action_planning import RDR2ActionPlanningProvider, ActionPlanningProvider, StardewActionPlanningProvider
from .module.task_inference import RDR2TaskInferenceProvider, TaskInferenceProvider, StardewTaskInferenceProvider
from .module.skill_curation import RDR2SkillCurationProvider, SkillCurationProvider

from .execute.skill_execute import SkillExecuteProvider

from .augment.draw_mask_panel import DrawMaskPanelProvider
from .augment.draw_grids import DrawGridsProvider
from .augment.draw_axis import DrawAxisProvider
from .augment.draw_color_band import DrawColorBandProvider

from .others.clip_minimap import ClipMinimapProvider
from .others.coordinates import CoordinatesProvider
from .others.task_guidance import TaskGuidanceProvider


__all__ = [
    # Base provider
    "BaseProvider",

    # LLM provider
    "LLMProvider",
    "EmbeddingProvider",
    "OpenAIProvider",
    "ClaudeProvider",
    "RestfulClaudeProvider",

    # Object detection provider
    "GdProvider",

    # Video provider
    "VideoEasyOCRExtractProvider",
    "VideoRecordProvider",
    "VideoFrameExtractProvider",
    "VideoClipProvider",

    # Module provider
    "RDR2InformationGatheringProvider",
    "RDR2SelfReflectionProvider",
    "RDR2ActionPlanningProvider",
    "RDR2TaskInferenceProvider",
    "RDR2SkillCurationProvider",
    "InformationGatheringProvider",
    "SelfReflectionProvider",
    "ActionPlanningProvider",
    "TaskInferenceProvider",
    "SkillCurationProvider",
    "StardewInformationGatheringProvider",
    "StardewSelfReflectionProvider",
    "StardewActionPlanningProvider",
    "StardewTaskInferenceProvider",

    # Augment provider
    "DrawMaskPanelProvider",
    "DrawGridsProvider",
    "DrawAxisProvider",
    "DrawColorBandProvider",

    # Others
    "ClipMinimapProvider",
    "CoordinatesProvider",
    "TaskGuidanceProvider",
]
