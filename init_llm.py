import chromadb
import os
from llama_index.llms.gemini import Gemini
from llama_index.core import (
    Document,
    VectorStoreIndex,
    StorageContext,
    Settings,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from common import *

def get_chroma_collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_or_create_collection(COLLECTION_NAME)


def get_vector_store_index() -> VectorStoreIndex:
    collection = get_chroma_collection()
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)


def ensure_llamaindex_settings():
    #Settings.llm = Ollama(model="llama3.2:latest", request_timeout=200, context_window=8000)
    Settings.llm = Ollama(model="gemma3:latest", request_timeout=200, context_window=4096)
    Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")
    Settings.node_parser = SentenceSplitter(chunk_size=300, chunk_overlap=100)


def ensure_llamaindex_agent_settings():
    GOOGLE_API_KEY = "AIzaSyDA58U7c2J6VdWdojDP7fqG8WGU8oVn8h4"  # add your GOOGLE API key here
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    llm = Gemini(
        model="models/gemini-2.5-flash",
    )
    Settings.llm = llm
    Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")
    Settings.node_parser = SentenceSplitter(chunk_size=300, chunk_overlap=100)