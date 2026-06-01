from src.vector.adapter import VectorStoreAdapter

class QdrantStore(VectorStoreAdapter):
    def __init__(self, url: str, api_key: str):
        self.url = url
        self.api_key = api_key
        # Placeholder initialization

    def upsert(self, id: str, text: str, metadata: dict) -> None:
        # Placeholder upsert logic
        pass

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        # Placeholder search logic
        return []

    def is_available(self) -> bool:
        # Placeholder availability check
        return False
