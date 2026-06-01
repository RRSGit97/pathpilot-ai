from abc import ABC, abstractmethod

class VectorStoreAdapter(ABC):
    @abstractmethod
    def upsert(self, id: str, text: str, metadata: dict) -> None:
        pass

    @abstractmethod
    def search(self, query: str, top_k: int = 5) -> list[dict]:
        pass

    @abstractmethod
    def is_available(self) -> bool:
        pass
