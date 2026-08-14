#!/usr/bin/env python
import importlib.util
from pathlib import Path
from types import SimpleNamespace

MODULE_PATH = Path(__file__).parents[1] / "memrl/service/mem0_memory_service.py"
spec = importlib.util.spec_from_file_location("mem0_memory_service_under_test", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
Mem0MemoryService = module.Mem0MemoryService


class FakeEmbeddings:
    def __init__(self):
        self.calls = []

    def create(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return SimpleNamespace(data=[SimpleNamespace(embedding=[1.0], index=0)])


class FakeEmbedder:
    def __init__(self):
        self.client = SimpleNamespace(embeddings=FakeEmbeddings())
        self.config = SimpleNamespace(model="Qwen3-Embedding-8B", embedding_dims=4096)

    def embed(self, text, memory_action=None):
        return self.client.embeddings.create(
            input=[text], model=self.config.model,
            dimensions=self.config.embedding_dims,
        ).data[0].embedding

    def embed_batch(self, texts, memory_action="add"):
        return self.client.embeddings.create(
            input=texts, model=self.config.model,
            dimensions=self.config.embedding_dims,
        ).data


def main():
    service = Mem0MemoryService.__new__(Mem0MemoryService)
    service._omit_embedding_dimensions = True
    service.memory = SimpleNamespace(embedding_model=FakeEmbedder())
    service._mem0_config = {
        "vector_store": {"config": {"embedding_model_dims": 4096}}
    }

    service._configure_embedding_client()
    wrapper = service.memory.embedding_model.client.embeddings.create
    service._configure_embedding_client()
    assert service.memory.embedding_model.client.embeddings.create is wrapper

    # Covers ordinary search, entity boost (also embed(..., "search")), and batch add.
    service.memory.embedding_model.embed("main query", "search")
    service.memory.embedding_model.embed("entity text", "search")
    service.memory.embedding_model.embed_batch(["a", "b"], "add")

    calls = service.memory.embedding_model.client.embeddings.calls
    assert len(calls) == 3
    assert all("dimensions" not in kwargs for _, kwargs in calls)
    assert service._mem0_config["vector_store"]["config"]["embedding_model_dims"] == 4096
    print("OK: dimensions stripped for main/entity/batch; Qdrant dims remain 4096")


if __name__ == "__main__":
    main()
