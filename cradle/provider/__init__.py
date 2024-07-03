from .base import BaseProvider
from .base.base_provider import BaseModuleProvider
from .base.base_embedding import EmbeddingProvider
from .base.base_llm import LLMProvider

from .llm.openai import OpenAIProvider
from .llm.claude import ClaudeProvider
from .llm.restful_claude import RestfulClaudeProvider

from .circle_detector import CircleDetectProvider
from .sam_provider import SamProvider
from .object_detect.gd_provider import GdProvider

from .video.video_ocr_extractor import VideoOCRExtractorProvider
from .video.video_recorder import VideoRecordProvider
from .video.video_frame_extractor import VideoFrameExtractorProvider
from .video.video_clip import VideoClipProvider

from .process.action_planning import (ActionPlanningPreprocessProvider,
                                      ActionPlanningPostprocessProvider,
                                      RDR2ActionPlanningPreprocessProvider,
                                      RDR2ActionPlanningPostprocessProvider,
                                      StardewActionPlanningPreprocessProvider,
                                      StardewActionPlanningPostprocessProvider)

from .process.information_gathering import (InformationGatheringPreprocessProvider,
                                            InformationGatheringPostprocessProvider,
                                            RDR2InformationGatheringPreprocessProvider,
                                            RDR2InformationGatheringPostprocessProvider,
                                            StardewInformationGatheringPreprocessProvider,
                                            StardewInformationGatheringPostprocessProvider)

from .process.self_reflection import (SelfReflectionPreprocessProvider,
                                      SelfReflectionPostprocessProvider,
                                      RDR2SelfReflectionPostprocessProvider,
                                      RDR2SelfReflectionPreprocessProvider,
                                      StardewSelfReflectionPreprocessProvider,
                                      StardewSelfReflectionPostprocessProvider)

from .process.task_inference import (TaskInferencePreprocessProvider,
                                     TaskInferencePostprocessProvider,
                                     RDR2TaskInferencePreprocessProvider,
                                     RDR2TaskInferencePostprocessProvider,
                                     StardewTaskInferencePreprocessProvider,
                                     StardewTaskInferencePostprocessProvider)

from .module.information_gathering import RDR2InformationGatheringProvider, InformationGatheringProvider, StardewInformationGatheringProvider
from .module.self_reflection import RDR2SelfReflectionProvider, SelfReflectionProvider, StardewSelfReflectionProvider
from .module.action_planning import RDR2ActionPlanningProvider, ActionPlanningProvider, StardewActionPlanningProvider
from .module.task_inference import RDR2TaskInferenceProvider, TaskInferenceProvider, StardewTaskInferenceProvider
from .module.skill_curation import RDR2SkillCurationProvider, SkillCurationProvider

from .execute.skill_execute import SkillExecuteProvider

from .augment.augment import AugmentProvider

from .others.coordinates import CoordinatesProvider
from .others.task_guidance import TaskGuidanceProvider


__all__ = [
    # Base provider
    "BaseProvider",

    # LLM providers
    "LLMProvider",
    "EmbeddingProvider",
    "OpenAIProvider",
    "ClaudeProvider",
    "RestfulClaudeProvider",

    # Object detection provider
    "GdProvider",

    # Video provider
    "VideoOCRExtractorProvider",
    "VideoRecordProvider",
    "VideoFrameExtractorProvider",
    "VideoClipProvider"

    # Augmentation providers
    "AugmentProvider",

    # Others
    "CoordinatesProvider",
    "TaskGuidanceProvider",

    # ???
    "CircleDetectProvider",
    "SamProvider",

    # Process provider
    "SkillExecuteProvider"
    "ActionPlanningPreprocessProvider",
    "ActionPlanningPostprocessProvider",
    "RDR2ActionPlanningPreprocessProvider",
    "RDR2ActionPlanningPostprocessProvider",
    "StardewActionPlanningPreprocessProvider",
    "StardewActionPlanningPostprocessProvider",
    "InformationGatheringPreprocessProvider",
    "InformationGatheringPostprocessProvider",
    "RDR2InformationGatheringPreprocessProvider",
    "RDR2InformationGatheringPostprocessProvider",
    "StardewInformationGatheringPreprocessProvider",
    "StardewInformationGatheringPostprocessProvider",
    "SelfReflectionPreprocessProvider",
    "SelfReflectionPostprocessProvider",
    "RDR2SelfReflectionPostprocessProvider",
    "RDR2SelfReflectionPreprocessProvider",
    "StardewSelfReflectionPreprocessProvider",
    "StardewSelfReflectionPostprocessProvider",
    "TaskInferencePreprocessProvider",
    "TaskInferencePostprocessProvider",
    "RDR2TaskInferencePreprocessProvider",
    "RDR2TaskInferencePostprocessProvider",
    "StardewTaskInferencePreprocessProvider",
    "StardewTaskInferencePostprocessProvider",

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
]
