"""Register the text-only Modujo Qwen4-Exp checkpoint with ms-swift."""

from swift.model import Model, ModelGroup, ModelMeta, register_model
from swift.model.model_arch import ModelArch
from swift.model.register import ModelLoader
from swift.template import TemplateType


MODEL_PATH = "/root/.cache/huggingface/hub/models--Alexhu1999--Modujo-9B-A1B/snapshots/a18b15449fbce9d4e182138704d8d015f2dadfbf"


register_model(
    ModelMeta(
        "modujo_qwen4_exp",
        [ModelGroup([Model(hf_model_id="Alexhu1999/Modujo-9B-A1B", model_path=MODEL_PATH)])],
        ModelLoader,
        # The text checkpoint uses the Qwen3.8 tokenizer, but the qwen3_8
        # Swift template is multimodal and expects get_rope_index().
        template=TemplateType.qwen3,
        model_arch=ModelArch.llama,
        architectures=["Qwen4ExpForCausalLM"],
        requires=["transformers>=5.16.0"],
        tags=["text-generation", "moe"],
    ),
    exist_ok=True,
)
