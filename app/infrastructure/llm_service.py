from langchain_community.llms import LlamaCpp
from . import config

class LLMService:
    def __init__(self, model_path: str):
        self.llm = LlamaCpp(
            model_path=model_path,
            n_gpu_layers=12,
            n_batch=512,
            n_ctx=4096,
            f16_kv=True,
            verbose=True,
        )

    def get_llm(self):
        return self.llm

# Singleton instance
llm_service = LLMService(model_path=config.MODEL_PATH)