
import os
import uuid
from typing import Dict, Any, Optional, List, Tuple

# LlamaIndex imports (v0.10+)
from llama_index.core import (
    Document,
    VectorStoreIndex,
    StorageContext,
    Settings,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter

# Ollama-backed LLM & Embeddings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

# Chroma vector store
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from common import *
from init_llm import *
from db import *


# ---------- Ingestion & Reindex ----------

def write_text_cache(candidate_id: str, text: str):
    path = os.path.join(TEXT_DIR, f"{candidate_id}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def read_text_cache(candidate_id: str) -> Optional[str]:
    path = os.path.join(TEXT_DIR, f"{candidate_id}.txt")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return None


def ingest_candidate_text(candidate_id: str, text: str, metadata: Dict[str, Any]):
    ensure_llamaindex_settings()
    # Create a Document with metadata filterable by candidate_id
    doc = Document(
        text=text,
        metadata={
            "candidate_id": candidate_id,
            "name": metadata.get("name") or "",
            "profession": metadata.get("profession") or "",
            "years_experience": metadata.get("years_experience") if metadata.get("years_experience") is not None else "",
        },
        doc_id=f"cv::{candidate_id}::{uuid.uuid4().hex}",  # unique chunk-batch id to support reindex
    )
    collection = get_chroma_collection()
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    # Add to vector store (this call will upsert nodes)
    VectorStoreIndex.from_documents([doc], storage_context=storage_context, show_progress=True)



# ---------- Retrieval & QA ----------
def summarize_candidate(candidate_id: str, name: str) -> str:
    qe = get_query_engine_for_candidate(candidate_id, k=6)
    prompt = (
        "You are summarizing a candidate CV for a recruiter. "
        "Summarize in 6-10 bullet points: core skills, seniority, domains, notable projects, leadership, tools, and education. "
        "Use concise, recruiter-friendly wording."
    )
    try:
        resp = qe.query(f"{prompt}\n\nCandidate: {name}. Provide a concise, objective summary.")
        return str(resp)
    except Exception as e:
        return f"Summary unavailable: {e}"



def get_query_engine_for_candidate(candidate_id: str, k: int = 5):
    ensure_llamaindex_settings()
    index = get_vector_store_index()
    filters = MetadataFilters(filters=[ExactMatchFilter(key="candidate_id", value=candidate_id)])
    return index.as_query_engine(similarity_top_k=k, filters=filters)

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

from typing import Tuple, List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter


def retrieve_vector_data(candidate_id: str, question: str, k: int = 5) -> List[Any]:
    """
    Retrieve relevant nodes from the vector store for a given candidate and question.
    
    Args:
        candidate_id: The candidate identifier to filter by
        question: The question to search for
        k: Number of top similar results to retrieve
        
    Returns:
        List of retrieved nodes
    """
    index = get_vector_store_index()
    filters = MetadataFilters(filters=[ExactMatchFilter(key="candidate_id", value=candidate_id)])
    retriever = index.as_retriever(similarity_top_k=k, filters=filters)
    
    log(f"[ret] retrieving for candidate {candidate_id} question='{question}'")
    nodes = retriever.retrieve(question)
    
    return nodes


def process_retrieved_nodes(nodes: List[Any], candidate_id: str, k: int = 5) -> List[Dict[str, Any]]:
    """
    Process retrieved nodes and extract relevant information into a structured format.
    
    Args:
        nodes: List of retrieved nodes from vector store
        candidate_id: The candidate identifier for logging
        
    Returns:
        List of processed source dictionaries
    """
    sources: List[Dict[str, Any]] = []
    
    for s in nodes:
        node = getattr(s, "node", s)
        meta = getattr(node, "metadata", getattr(s, "metadata", {})) or {}
        
        # Safe text extraction
        try:
            text = node.get_content() if hasattr(node, "get_content") else getattr(node, "text", "") or ""
        except Exception:
            text = getattr(node, "text", "") or ""
        
        item = {
            "score": getattr(s, "score", None),
            "node_id": getattr(node, "node_id", getattr(s, "id_", None)),
            "metadata": meta,
            "text": text,
            "text_preview": text[:200].replace("\n", " "),
        }
        sources.append(item)
        
        # Log retrieved item details
        log(f"[ret] cand={candidate_id} score={item['score']} id={item['node_id']} "
            f"meta={meta} "
            f"prev='{item['text'][:500]}...'")
    
    return sources


def build_context_prompt(question: str, sources: List[Dict[str, Any]]) -> str:
    """
    Build a context-augmented prompt using the retrieved sources.
    
    Args:
        question: The original question
        sources: List of retrieved source documents
        
    Returns:
        Formatted prompt with context
    """
    if not sources:
        return question
    
    context_parts = []
    for i, source in enumerate(sources, 1):
        context_parts.append(f"Source {i}:\n{source['text']}\n")
    
    context = "\n".join(context_parts)
    
    prompt = f"""Based on the following context, please answer the question.

Context:
{context}

Question: {question}

Answer:"""
    
    return prompt


def generate_llm_answer_with_context(candidate_id: str, context_prompt: str, timeout_s: int = 20) -> str:
    """
    Generate an LLM answer using the context-augmented prompt with timeout protection.
    
    Args:
        candidate_id: The candidate identifier
        context_prompt: The question augmented with retrieved context
        timeout_s: Timeout in seconds for generation
        
    Returns:
        Generated answer string, or fallback message if generation fails
    """
    answer = "[retrieval-only]"
    
    try:
        # Use a basic LLM instead of query engine to ensure we control the context
        llm = Settings.llm
        
        with ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(llm.complete, context_prompt)
            resp = fut.result(timeout=timeout_s)
        answer = str(resp)
    except FuturesTimeout:
        log(f"LLM generation timed out after {timeout_s}s; returning retrieval-only.")
    except Exception as e:
        log(f"LLM generation failed: {e}")
    
    return answer


# Alternative implementation if you want to use query engine but with specific nodes
def generate_llm_answer_with_nodes(candidate_id: str, question: str, nodes: List[Any], timeout_s: int = 20) -> str:
    """
    Generate an LLM answer using the specific retrieved nodes.
    
    Args:
        candidate_id: The candidate identifier
        question: The original question
        nodes: The specific nodes to use as context
        timeout_s: Timeout in seconds for generation
        
    Returns:
        Generated answer string, or fallback message if generation fails
    """
    answer = "[retrieval-only]"
    
    try:
        from llama_index.core.query_engine import RetrieverQueryEngine
        from llama_index.core.retrievers import BaseRetriever
        
        # Create a custom retriever that returns our specific nodes
        class PreRetrievedRetriever(BaseRetriever):
            def __init__(self, nodes):
                self._nodes = nodes
            
            def _retrieve(self, query_bundle):
                return self._nodes
        
        retriever = PreRetrievedRetriever(nodes)
        query_engine = RetrieverQueryEngine(retriever=retriever)
        
        with ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(query_engine.query, question)
            resp = fut.result(timeout=timeout_s)
        answer = str(resp)
    except FuturesTimeout:
        log(f"LLM generation timed out after {timeout_s}s; returning retrieval-only.")
    except Exception as e:
        log(f"LLM generation failed: {e}")
    
    return answer


def rag_answer(candidate_id: str, question: str, k: int = 5, gen_timeout_s: int = 20) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Proper RAG pipeline: Retrieve relevant data, augment with context, generate answer.
    
    Args:
        candidate_id: The candidate identifier to filter by
        question: The question to answer
        k: Number of top similar results to retrieve
        gen_timeout_s: Timeout in seconds for LLM generation
        
    Returns:
        Tuple of (generated_answer, list_of_sources)
    """
    ensure_llamaindex_settings()
    # Step 1: Retrieve data from vectors
    nodes = retrieve_vector_data(candidate_id, question, k)
    
    # Step 2: Process retrieved nodes into structured sources
    sources = process_retrieved_nodes(nodes, candidate_id)
    
    # Step 3: Generate answer using the retrieved context
    # Option A: Build context prompt manually
    #context_prompt = build_context_prompt(question, sources)
    #log(f"Context-augmented prompt:\n{context_prompt}\n--- End of prompt ---")
    
    #answer = generate_llm_answer_with_context(candidate_id, context_prompt, gen_timeout_s)
    
    # Option B: Use nodes directly with query engine (uncomment to use instead)
    answer = generate_llm_answer_with_nodes(candidate_id, question, nodes, gen_timeout_s)
    
    return answer, sources



def remove_candidate(candidate_id: str):
    delete_candidate_vectors(candidate_id)
    db_execute("DELETE FROM candidates WHERE id=?", (candidate_id,))
    # Remove cached text
    path = os.path.join(TEXT_DIR, f"{candidate_id}.txt")
    if os.path.exists(path):
        try:
            os.remove(path)
        except Exception:
            pass


def delete_candidate_vectors(candidate_id: str):
    # Directly purge from Chroma by metadata filter (Chroma API)
    collection = get_chroma_collection()
    try:
        collection.delete(where={"candidate_id": candidate_id})
    except Exception:
        pass