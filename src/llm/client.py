from src.config.settings import settings

class LLMClient:
    def __init__(self):
        self.provider = settings.LLM_PROVIDER
        # Placeholder initialization

    def generate(self, prompt: str, system_instruction: str = None) -> str:
        # Placeholder generation logic
        return ""
