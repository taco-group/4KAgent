"""Conftest for 4KAgent tests — sets up sys.path and mocks missing dependencies."""

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

# Ensure project root is first on the path
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Remove any previously cached 'utils' module that's not ours
for mod_name in list(sys.modules.keys()):
    if mod_name == 'utils' or mod_name.startswith('utils.'):
        mod = sys.modules[mod_name]
        if mod is not None and hasattr(mod, '__file__') and mod.__file__:
            if PROJECT_ROOT not in str(mod.__file__):
                del sys.modules[mod_name]

# Create stub modules for heavy dependencies not needed in unit tests
_stub_modules = [
    'utils', 'utils.misc', 'utils.logger', 'utils.custom_types',
    'utils.img_tree', 'utils.restore_profile', 'utils.expert_IQA_eval',
    'utils.expert_face_score', 'utils.color_fix', 'utils.scorer',
    'pipeline', 'pipeline.prompts', 'pipeline.profile_loader',
    'pipeline.the4kagent_pipeline',
    'executor', 'executor.executor',
    'facexlib', 'facexlib.utils', 'facexlib.utils.face_restoration_helper',
    'openai', 'llamaapi',
    'torch', 'PIL', 'outlines', 'outlines.models',
    'outlines.models.transformers_vision', 'outlines.generate',
    'transformers', 'pydantic', 'hpsv2',
]

for name in _stub_modules:
    if name not in sys.modules:
        mod = types.ModuleType(name)
        # Make parent packages aware of child stubs
        parts = name.split('.')
        if len(parts) > 1:
            parent_name = '.'.join(parts[:-1])
            if parent_name in sys.modules:
                setattr(sys.modules[parent_name], parts[-1], mod)
        sys.modules[name] = mod

# Populate commonly needed stubs
sys.modules['utils.misc'].encode_img = MagicMock(return_value="data:image/jpeg;base64,stub")
sys.modules['utils.misc'].sorted_glob = MagicMock(return_value=[])
sys.modules['utils.logger'].get_logger = MagicMock(return_value=MagicMock())
sys.modules['utils.custom_types'].Degradation = str
sys.modules['utils.custom_types'].Level = str
sys.modules['utils.custom_types'].Subtask = str
sys.modules['utils.custom_types'].ToolName = str

# pipeline.prompts stubs
_prompts = sys.modules['pipeline.prompts']
_prompts.depictqa_evaluate_degradation_prompt = ""
_prompts.depictqa_compare_prompt = ""
_prompts.system_message = ""
_prompts.updated_perception_system_message = ""
_prompts.gpt_evaluate_degradation_prompt = ""
_prompts.gpt_evaluate_tool_result_prompt = ""
_prompts.gpt_compare_prompt = ""
_prompts.schedule_w_retrieval_prompt = ""
_prompts.schedule_wo_retrieval_prompt = ""
_prompts.reason_to_schedule_prompt = ""
_prompts.schedule_updated_w_retrieval_prompt = ""
_prompts.reschedule_ps_prompt = ""
_prompts.updated_perception_system_prompt = ""
_prompts.llama_vision_agent_perception_no_brighten_system_message = ""
_prompts.llama_vision_agent_plan_system_message = ""

# openai stubs
sys.modules['openai'].AzureOpenAI = MagicMock
sys.modules['llamaapi'].LlamaAPI = MagicMock

# transformers stubs
sys.modules['transformers'].MllamaForConditionalGeneration = MagicMock
sys.modules['transformers'].Qwen2_5_VLForConditionalGeneration = MagicMock
sys.modules['transformers'].AutoProcessor = MagicMock

# outlines stubs
sys.modules['outlines.models.transformers_vision'].transformers_vision = MagicMock
sys.modules['outlines.generate'].json = MagicMock

# pydantic stubs
sys.modules['pydantic'].BaseModel = type

# facexlib stubs
sys.modules['facexlib.utils.face_restoration_helper'].FaceRestoreHelper = MagicMock

# executor stubs
sys.modules['executor'].executor = MagicMock

# PIL stubs
sys.modules['PIL'].Image = MagicMock

# torch stubs
_torch = sys.modules['torch']
_torch.no_grad = lambda: lambda fn: fn
_torch.bfloat16 = None
_torch.cuda = MagicMock

# utils.restore_profile — wildcard import gives names
for name in [
    'General_Baseline_Brightening_ToolName',
    'General_Baseline_Defocus_Deblurring_ToolName',
    'General_Baseline_Dehazing_ToolName',
    'General_Baseline_Denoising_ToolName',
    'General_Baseline_Deraining_ToolName',
    'General_Baseline_Jpeg_Compression_Artifact_Removal_ToolName',
    'General_Baseline_Motion_Deblurring_ToolName',
    'General_Baseline_SuperResolution_ToolName',
    'General_SuperResolution_Fidelity_ToolName',
    'General_SuperResolution_Perception_ToolName',
    'General_SuperResolution_2x_Fidelity_ToolName',
    'General_SuperResolution_2x_Perception_ToolName',
]:
    setattr(sys.modules['utils.restore_profile'], name, [])

# utils.color_fix
for fn in ['adain_color_fix', 'wavelet_color_fix', 'cv2_to_pil', 'pil_to_cv2']:
    setattr(sys.modules['utils.color_fix'], fn, MagicMock())

# utils.expert_IQA_eval
sys.modules['utils.expert_IQA_eval'].compute_iqa = MagicMock()
sys.modules['utils.expert_IQA_eval'].compute_iqa_metric_score = MagicMock()
sys.modules['utils.expert_IQA_eval'].compute_iqa_metric_score_batch = MagicMock()

# utils.expert_face_score
sys.modules['utils.expert_face_score'].compute_face_scores = MagicMock()

# utils.scorer
sys.modules['utils.scorer'].calculate_cos_dist = MagicMock()
sys.modules['utils.scorer'].calculate_niqe = MagicMock()

# pipeline.profile_loader
sys.modules['pipeline.profile_loader'].load_profile_config = MagicMock(return_value={})
