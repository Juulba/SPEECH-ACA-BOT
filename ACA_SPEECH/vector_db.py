import os
import threading
import uuid

from pinecone import Pinecone, ServerlessSpec
from setup.keys import HF_TOKEN, PINECONE_KEYS

if HF_TOKEN:
    # Keep auth local and cheap at startup; avoid network login during import.
    os.environ.setdefault("HF_TOKEN", HF_TOKEN)
    os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", HF_TOKEN)
    os.environ.setdefault("HUGGINGFACE_TOKEN", HF_TOKEN)


class VectorDB:
    _shared_model = None
    _shared_model_lock = threading.Lock()

    _shared_pinecone = None
    _shared_index = None
    _shared_index_lock = threading.Lock()

    _segments_initialized = False
    _segments_lock = threading.Lock()

    def __init__(self, clear_on_init=False):
        self.model = self._get_model()
        self.pinecone, self.vector_db = self._get_vector_db()

        if clear_on_init:
            self.clear_db()

    @classmethod
    def _get_vector_db(cls):
        if cls._shared_pinecone is not None and cls._shared_index is not None:
            return cls._shared_pinecone, cls._shared_index

        with cls._shared_index_lock:
            if cls._shared_pinecone is None or cls._shared_index is None:
                api_key = PINECONE_KEYS["secret_key"]
                index_name = PINECONE_KEYS["key_name"]

                pc = Pinecone(api_key=api_key)

                if index_name not in pc.list_indexes().names():
                    pc.create_index(
                        name=index_name,
                        dimension=cls._shared_model.get_sentence_embedding_dimension(),
                        metric="cosine",
                        spec=ServerlessSpec(
                            cloud="aws",
                            region="us-east-1",
                        ),
                    )

                cls._shared_pinecone = pc
                cls._shared_index = pc.Index(name=index_name)

        return cls._shared_pinecone, cls._shared_index

    @classmethod
    def _get_model(cls):
        if cls._shared_model is not None:
            return cls._shared_model

        with cls._shared_model_lock:
            if cls._shared_model is None:
                from sentence_transformers import SentenceTransformer

                cls._shared_model = SentenceTransformer(
                    "all-MiniLM-L6-v2",
                    cache_folder="/Users/JulieB/.hf_cache",
                )

        return cls._shared_model

    def ensure_segments(self, segments):
        if VectorDB._segments_initialized:
            return

        with VectorDB._segments_lock:
            if VectorDB._segments_initialized:
                return

            self.store_segments(segments)
            VectorDB._segments_initialized = True

    def init_vector_db(self):
        api_key = PINECONE_KEYS["secret_key"]
        index_name = PINECONE_KEYS["key_name"]

        pc = Pinecone(api_key=api_key)

        if index_name not in pc.list_indexes().names():
            pc.create_index(
                name=index_name,
                dimension=self.model.get_sentence_embedding_dimension(),
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )

        self.pinecone = pc
        return pc.Index(name=index_name)

    def vectorize(self, text_segments):
        # Batch embed
        embeddings = self.model.encode(
            text_segments,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,  # good for cosine similarity
        )

        vectors = []
        for segment, emb in zip(text_segments, embeddings):
            vid = str(uuid.uuid4())  # unique id so you don't overwrite old vectors
            vectors.append({
                "id": vid,
                "values": emb.tolist(),
                "metadata": {"text": segment},  # store text inside Pinecone
            })

        return vectors

    def store_vectors(self, vectors):
        self.vector_db.upsert(vectors)

    def store_segments(self, text_segments):
        vectors = self.vectorize(text_segments)
        self.store_vectors(vectors)

    def clear_db(self):
        try:
            index_name = PINECONE_KEYS["key_name"]
            self.pinecone.delete_index(index_name)
            self.vector_db = self.init_vector_db()
        except Exception as e:
            print("Not able to delete index:", e)

    def query_db(self, query, top_k):
        query_vector = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )[0].tolist()

        results = self.vector_db.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True
        )

        return [
            m["metadata"]["text"]
            for m in results["matches"]
            if "metadata" in m and "text" in m["metadata"]
        ]

