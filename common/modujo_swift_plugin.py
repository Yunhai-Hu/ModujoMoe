"""Register the text-only Modujo Qwen4-Exp checkpoint with ms-swift."""

from swift.model import Model, ModelGroup, ModelMeta, register_model
from swift.model.model_arch import ModelArch
from swift.model.register import ModelLoader
from swift.template import TemplateType


register_model(
    ModelMeta(
        "modujo_qwen4_exp",
        [ModelGroup([Model(hf_model_id="Alexhu1999/Modujo-9B-A1B")])],
        ModelLoader,
        # qwen3_8 is a multimodal Swift template that expects get_rope_index().
        # This checkpoint is text-only, so the compatible Qwen3 template is used.
        template=TemplateType.qwen3,
        model_arch=ModelArch.llama,
        architectures=["Qwen4ExpForCausalLM"],
        requires=["transformers>=5.16.0"],
        tags=["text-generation", "moe"],
    ),
    exist_ok=True,
)
