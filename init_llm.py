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
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from common import *


def get_chroma_collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_or_create_collection(COLLECTION_NAME)

def get_vector_store_index() -> VectorStoreIndex:
    collection = get_chroma_collection()
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store
    )
    return VectorStoreIndex.from_vector_store(
        vector_store,
        storage_context=storage_context,
        embed_model=Settings.embed_model,
    )

def ensure_openAI_settings():
    #os.environ["OPENAI_API_KEY"] =  ""

    Settings.llm = OpenAI(
        model="gpt-4o-mini",       # fast + cheap (for RAG/chat)
        temperature=0.2
    )

    Settings.embed_model = OpenAIEmbedding(
        model="text-embedding-3-small"   # or "text-embedding-3-small" for cheaper use
    )

def ensure_ollama_settings():
    #Settings.llm = Ollama(model="llama3.2:latest", request_timeout=200, context_window=8000)
    Settings.llm = Ollama(model="gemma3:latest", request_timeout=200, context_window=4096)
    Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")
    Settings.node_parser = SentenceSplitter(chunk_size=300, chunk_overlap=100)
    
    
def ensure_gemini_settings():
    GOOGLE_API_KEY = ""  # add your GOOGLE API key here
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    llm = Gemini(
        model="models/gemini-2.5-flash",
    )
    Settings.llm = llm
    Settings.embed_model = GoogleGenAIEmbedding(
        model="models/text-embedding-004",
    )
    Settings.node_parser = SentenceSplitter(chunk_size=300, chunk_overlap=100)
    
def ensure_llamaindex_settings():
    ensure_openAI_settings()
    #ensure_gemini_settings()


def ensure_llamaindex_agent_settings():
    GOOGLE_API_KEY = ""  # add your GOOGLE API key here
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    llm = Gemini(
        model="models/gemini-2.5-flash",
    )
    Settings.llm = llm
    Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")
    Settings.node_parser = SentenceSplitter(chunk_size=300, chunk_overlap=100)
    
    
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
    
from llama_index.llms.bedrock import Bedrock
from llama_index.embeddings.bedrock import BedrockEmbedding
from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader


def ensure_gen_engine_openai():
    Settings.llm = OpenAI(
        model="gemini-2.5-flash",
        api_base=os.environ.get("GEN_ENGINE_API_BASE", ""),
        api_key=os.environ.get("GEN_ENGINE_API_KEY", ""),
        temperature=0.2,
        max_tokens=1024,
    )

    Settings.embed_model = OpenAIEmbedding(
        model_name="text-embedding-ada-002",
        api_base=os.environ.get("GEN_ENGINE_API_BASE", ""),
        api_key=os.environ.get("GEN_ENGINE_API_KEY", ""),
    )
