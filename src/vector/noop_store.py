from src.vector.adapter import VectorStoreAdapter

class NoopStore(VectorStoreAdapter):
    def upsert(self, id: str, text: str, metadata: dict) -> None:
        pass

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        return []

    def is_available(self) -> bool:
        return False
