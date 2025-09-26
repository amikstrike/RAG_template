
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

# Chroma vector store
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from common import *
from init_llm import *
from db import *

from typing import List, Optional
from structures import CandidateSummary

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

from llama_index.core.node_parser import SimpleNodeParser

def ingest_candidate_text(candidate_id: str, text: str, metadata: Dict[str, Any]):
    ensure_llamaindex_settings()
    os.makedirs(PERSIST_DIR, exist_ok=True)

    parser = SentenceSplitter(chunk_size=1000, chunk_overlap=100)
    nodes = parser.get_nodes_from_documents([
        Document(
            text=text,
            metadata={
                "candidate_id": candidate_id,
                "name": metadata.get("name") or "",
                "profession": metadata.get("profession") or "",
                "years_experience": metadata.get("years_experience") or "",
            },
            doc_id=f"cv::{candidate_id}::{uuid.uuid4().hex}",
        )
    ])

    collection = get_chroma_collection()
    vector_store = ChromaVectorStore(chroma_collection=collection)

    storage_context = StorageContext.from_defaults(
        vector_store=vector_store
    )

    index = VectorStoreIndex(
        nodes,
        storage_context=storage_context,
        embed_model=Settings.embed_model,
        show_progress=True,
        insert_batch_size=1,
    )
    
    return index

# ---------- Retrieval & QA ----------


from typing import cast

from llama_index.core.output_parsers import PydanticOutputParser
from llama_index.llms.openai import OpenAI

# your CandidateSummary schema (Pydantic)
from structures import CandidateSummary

from typing import Any
from llama_index.core import VectorStoreIndex
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
from llama_index.llms.openai import OpenAI

def get_query_engine_for_candidate(candidate_id: str, k: int = 6):
    ensure_llamaindex_settings()
    index: VectorStoreIndex = get_vector_store_index()
    
    collection = get_chroma_collection()
    log(f"[DEBUG] Chroma collection count: {collection.count()}")
    # Build filters
    filters = MetadataFilters(filters=[ExactMatchFilter(key="candidate_id", value=candidate_id)])

    # Debug: log filter
    log(f"[DEBUG] Using filter: key='candidate_id', value='{candidate_id}' (type={type(candidate_id)})")

    # Debug: check if any docs exist with that metadata
    try:
        docs = index.docstore.docs
        sample_docs = list(docs.values())[:3]
        log(f"[DEBUG] Total docs in index: {len(docs)}")
        for d in sample_docs:
            log(f"[DEBUG] Sample doc metadata: {d.metadata}")
    except Exception as e:
        log(f"[DEBUG] Could not inspect docstore: {e}")

    # Create structured LLM
    llm = Settings.llm.as_structured_llm(output_cls=CandidateSummary)

    # Try query engine with and without filters
    try:
        qe = index.as_query_engine(
            similarity_top_k=k,
            filters=filters,
            llm=llm,
        )
        log("[DEBUG] Query engine created successfully with filters.")
        return qe
    except Exception as e:
        log(f"[DEBUG] Failed to create query engine with filters: {e}")

        # fallback without filters
        qe = index.as_query_engine(
            similarity_top_k=k,
            llm=llm,
        )
        log("[DEBUG] Query engine created successfully WITHOUT filters.")
        return qe


def summarize_candidate(candidate_id: str, name: str) -> CandidateSummary | str:
    qe = get_query_engine_for_candidate(candidate_id, k=6)

    prompt = (
        "You are summarizing a candidate CV for a recruiter.\n"
        "Return the result that fits the CandidateSummary schema. "
        "If something is unknown, leave list empty or field null.\n"
        f"Candidate: {name}"
    )

    try:
        response = qe.query(prompt)
        log(f"[DEBUG] Raw response: {response}")
        return str(response)
    except Exception as e:
        return f"Summary unavailable: {e}"


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
    log(f"[ret] retrieved {len(nodes)} nodes")
    
    return nodes




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
    log(f"[ret] retrieved {len(nodes)} nodes")
    sources = [{"text": n.get_text(), "metadata": n.metadata} for n in nodes]
    log(f"[ret] prepared {len(sources)} sources")
    # Step 2: Build context-augmented prompt
    prompt = build_context_prompt(question, sources)
    log(f"[gen] built prompt with {prompt}")
    # Step 3: Generate answer with timeout
    llm = Settings.llm
    answer = "No answer"
    answer = llm.complete(prompt)
    result = answer.text.strip()
    log(f"[gen] generated answer with {result}")
    
    return result, sources



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