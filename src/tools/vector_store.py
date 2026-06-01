"""
src/tools/vector_store.py
-------------------------
Optional vector store adapter for PathPilot AI.

Wraps Qdrant to provide semantic search capabilities for:
- Job descriptions
- Resource catalogs
- Project notes
- Resumes

If VECTOR_STORE_ENABLED is false in settings, or if Qdrant is 
unavailable, this adapter degrades gracefully and acts as a No-Op.
This ensures the MVP remains fully functional without a vector database.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List, Optional

from src.config import settings

logger = logging.getLogger(__name__)

# The collections expected by PathPilot AI
COLLECTIONS = [
    "job_description_chunks",
    "resource_catalog_chunks",
    "project_notes",
    "resume_chunks",
]

# We conditionally import Qdrant and SentenceTransformers so the app doesn't
# crash if the optional dependencies aren't installed or enabled.
try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qmodels
except ImportError:
    QdrantClient = None
    qmodels = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


class VectorStoreAdapter:
    """
    Adapter for Qdrant. Handles initialization, upserts, and searches.
    If disabled or broken, all methods return empty/safe defaults.
    """

    def __init__(self):
        self.enabled = settings.vector_store_enabled
        self.client: Optional[QdrantClient] = None
        self._embedder = None
        self._vector_size = 384  # Default for all-MiniLM-L6-v2

        if self.enabled:
            if QdrantClient is None:
                logger.warning("QdrantClient is not installed. Disabling vector store.")
                self.enabled = False
                return

            try:
                # Initialize Qdrant client
                if settings.qdrant_api_key:
                    self.client = QdrantClient(
                        url=settings.qdrant_url,
                        api_key=settings.qdrant_api_key,
                    )
                else:
                    self.client = QdrantClient(url=settings.qdrant_url)
                logger.info("Qdrant client initialized successfully.")
            except Exception as e:
                logger.error("Failed to initialize QdrantClient: %s", e)
                self.enabled = False

    def _get_embedder(self):
        """Lazy load the sentence transformer model to save memory."""
        if self._embedder is None and SentenceTransformer is not None:
            logger.info("Loading SentenceTransformer model (all-MiniLM-L6-v2)...")
            # This is a small, fast model producing 384-d vectors
            self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
        return self._embedder

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        embedder = self._get_embedder()
        if embedder is None:
            logger.error("SentenceTransformer not available. Cannot generate embeddings.")
            return [[] for _ in texts]
        
        # encode() returns a numpy array, we convert to lists of floats
        embeddings = embedder.encode(texts)
        return embeddings.tolist()

    def init_collections(self) -> None:
        """
        Create the required collections if they do not exist.
        """
        if not self.enabled or not self.client:
            return

        try:
            existing_cols = [col.name for col in self.client.get_collections().collections]
            for col_name in COLLECTIONS:
                if col_name not in existing_cols:
                    logger.info("Creating Qdrant collection: %s", col_name)
                    self.client.create_collection(
                        collection_name=col_name,
                        vectors_config=qmodels.VectorParams(
                            size=self._vector_size,
                            distance=qmodels.Distance.COSINE,
                        ),
                    )
        except Exception as e:
            logger.error("Failed to initialize Qdrant collections: %s", e)
            self.enabled = False

    def clear_collections(self) -> None:
        """
        Drop all collections. Useful for development and test resets.
        """
        if not self.enabled or not self.client:
            return

        try:
            for col_name in COLLECTIONS:
                self.client.delete_collection(collection_name=col_name)
                logger.info("Deleted Qdrant collection: %s", col_name)
        except Exception as e:
            logger.error("Failed to clear Qdrant collections: %s", e)

    def upsert(
        self,
        collection_name: str,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        embeddings: Optional[List[List[float]]] = None,
    ) -> None:
        """
        Upsert documents into a Qdrant collection.
        If embeddings are not provided, they are generated automatically.
        """
        if not self.enabled or not self.client:
            return

        if collection_name not in COLLECTIONS:
            logger.error("Collection '%s' is not supported.", collection_name)
            return

        if not texts:
            return

        if metadatas is None:
            metadatas = [{} for _ in texts]

        if embeddings is None:
            embeddings = self.embed_texts(texts)

        try:
            points = []
            for text, meta, emb in zip(texts, metadatas, embeddings):
                if not emb:
                    continue  # Skip if embedding failed
                    
                # Ensure we have a valid UUID for Qdrant
                point_id = meta.get("id") or str(uuid.uuid4())
                
                # Qdrant requires IDs to be UUID strings or uint64. We enforce UUID strings.
                try:
                    uuid.UUID(str(point_id))
                except ValueError:
                    point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, str(point_id)))
                    
                payload = {**meta, "text": text}
                
                points.append(
                    qmodels.PointStruct(
                        id=point_id,
                        vector=emb,
                        payload=payload,
                    )
                )

            if points:
                self.client.upsert(
                    collection_name=collection_name,
                    points=points,
                )
                logger.info("Upserted %d points to %s", len(points), collection_name)
        except Exception as e:
            logger.error("Upsert failed for %s: %s", collection_name, e)

    def search(
        self,
        collection_name: str,
        query: str,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Search for the top-k most similar chunks to the query string.
        Returns a list of dicts: {"id": ..., "score": ..., "text": ..., "metadata": ...}
        """
        if not self.enabled or not self.client:
            return []

        try:
            query_vector = self.embed_texts([query])[0]
            if not query_vector:
                return []

            results = self.client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=limit,
            )

            formatted_results = []
            for res in results:
                payload = res.payload or {}
                text = payload.pop("text", "")
                formatted_results.append({
                    "id": res.id,
                    "score": res.score,
                    "text": text,
                    "metadata": payload,
                })
                
            return formatted_results
        except Exception as e:
            logger.error("Search failed for %s: %s", collection_name, e)
            return []


# Expose a singleton instance
vector_store = VectorStoreAdapter()
