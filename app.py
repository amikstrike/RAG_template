import os
import io
import uuid
import json
import time
import sqlite3
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple

import streamlit as st
from pypdf import PdfReader

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

from rag import *
from common import *
from db import *
from extraction import *
from tools import *


# ---------- Streamlit UI ----------

def ui_header():
    st.set_page_config(page_title="CV RAG Manager", page_icon="📄", layout="wide")
    st.title("📄 CV RAG Manager — LlamaIndex + Chroma + Ollama")
    st.caption("Upload PDFs, manage candidates, and run candidate-grounded Q&A locally.")

def render_ReAct_tab():
    st.subheader("! ReAct Agent")
    st.markdown("Agent workflow for CV management")

    query = st.text_input("Enter your query for the agent", key="react_query")
    if st.button("Run Agent") and query.strip():
        with st.spinner("Running agent..."):
            import asyncio
            response = asyncio.run(run_agent_sync(query))
        st.markdown("**Agent Response:**")
        st.write(response or "No response")
        st.success("Agent run complete.")
    

def render_manage_tab():
    st.subheader("📂 Manage CVs")
    st.markdown("Upload PDFs. We extract text, guess basic metadata, and index into Chroma.")

    uploaded = st.file_uploader("Upload CV PDFs", type=["pdf"], accept_multiple_files=True)
    if uploaded:
        for f in uploaded:
            file_bytes = f.read()
            raw_text = extract_text_from_pdf(file_bytes)
            meta_guess = simple_metadata_guess(raw_text, f.name)
            with st.expander(f"Ingest: {f.name}"):
                name = st.text_input("Name", value=meta_guess.get("name", ""), key=f"name_{f.name}")
                profession = st.text_input("Profession", value=meta_guess.get("profession", ""), key=f"prof_{f.name}")
                years = st.number_input("Years of experience", value=float(meta_guess.get("years_experience") or 0.0), min_value=0.0, step=0.5, key=f"years_{f.name}")
                notes = st.text_area("Notes (optional)", value="", key=f"notes_{f.name}")
                if st.button("➕ Add & Index", key=f"add_{f.name}"):
                    # Save file to disk for traceability (optional)
                    save_dir = os.path.join(DATA_DIR, "pdfs")
                    os.makedirs(save_dir, exist_ok=True)
                    dest_path = os.path.join(save_dir, f"{uuid.uuid4().hex}_{f.name}")
                    with open(dest_path, "wb") as out:
                        out.write(file_bytes)

                    candidate_id = add_or_update_candidate(None, name, profession, years, dest_path, notes)
                    write_text_cache(candidate_id, raw_text)
                    ingest_candidate_text(candidate_id, raw_text, {
                        "name": name,
                        "profession": profession,
                        "years_experience": years,
                    })
                    st.success(f"Indexed {name} (ID: {candidate_id[:8]}…)")

    st.divider()

    # Existing list
    st.markdown("### Existing Candidates")
    candidates = load_candidates()
    if not candidates:
        st.info("No candidates yet. Upload PDFs above.")
        return

    for c in candidates:
        col1, col2, col3, col4, col5 = st.columns([3, 2, 2, 2, 2])
        with col1:
            st.markdown(f"**{c['name']}**\n\n{c['profession'] or ''}")
        with col2:
            st.markdown(f"Years: **{c['years_experience'] if c['years_experience'] is not None else '—'}**")
        with col3:
            if st.button("👁 View", key=f"view_{c['id']}"):
                st.session_state["selected_candidate_id"] = c["id"]
                st.session_state["active_tab"] = "candidate"
        with col4:
            if st.button("✏️ Edit", key=f"edit_{c['id']}"):
                st.session_state["edit_candidate_id"] = c["id"]
        with col5:
            if st.button("🗑 Delete", key=f"del_{c['id']}"):
                remove_candidate(c["id"])
                st.warning(f"Deleted {c['name']}")
                st.rerun()

        # Inline editor
        if st.session_state.get("edit_candidate_id") == c["id"]:
            with st.expander(f"Edit {c['name']}", expanded=True):
                new_name = st.text_input("Name", value=c["name"], key=f"ename_{c['id']}")
                new_prof = st.text_input("Profession", value=c["profession"] or "", key=f"eprof_{c['id']}")
                new_years = st.number_input("Years of experience", value=float(c["years_experience"] or 0.0), min_value=0.0, step=0.5, key=f"eyears_{c['id']}")
                new_notes = st.text_area("Notes", value=c["notes"] or "", key=f"enotes_{c['id']}")
                replace_pdf = st.file_uploader("Replace PDF (optional)", type=["pdf"], key=f"ereplace_{c['id']}")
                if st.button("💾 Save", key=f"save_{c['id']}"):
                    dest_path = c["filepath"]
                    raw_text = read_text_cache(c["id"]) or ""
                    if replace_pdf is not None:
                        file_bytes = replace_pdf.read()
                        raw_text = extract_text_from_pdf(file_bytes)
                        # Save new file
                        save_dir = os.path.join(DATA_DIR, "pdfs")
                        os.makedirs(save_dir, exist_ok=True)
                        dest_path = os.path.join(save_dir, f"{uuid.uuid4().hex}_{replace_pdf.name}")
                        with open(dest_path, "wb") as out:
                            out.write(file_bytes)
                    # Update db
                    add_or_update_candidate(c["id"], new_name, new_prof, new_years, dest_path, new_notes)
                    # Reindex
                    if raw_text:
                        delete_candidate_vectors(c["id"])  # purge old chunks
                        write_text_cache(c["id"], raw_text)
                        ingest_candidate_text(c["id"], raw_text, {
                            "name": new_name, "profession": new_prof, "years_experience": new_years
                        })
                    st.success("Saved & reindexed.")
                    st.session_state["edit_candidate_id"] = None
                    st.rerun()

        st.markdown("---")


def render_candidate_tab():
    st.subheader("👤 Candidate")
    candidates = load_candidates()
    if not candidates:
        st.info("No candidates. Go to Manage CVs to add some.")
        return

    # Selection
    options = {f"{c['name']} — {c['profession'] or 'Unknown'} ({c['id'][:8]}…)": c for c in candidates}
    default_key = None
    if st.session_state.get("selected_candidate_id"):
        for k, v in options.items():
            if v["id"] == st.session_state["selected_candidate_id"]:
                default_key = k
                break
    selected_label = st.selectbox("Select candidate", list(options.keys()), index=list(options.keys()).index(default_key) if default_key else 0)
    selected = options[selected_label]
    st.session_state["selected_candidate_id"] = selected["id"]

    c1, c2 = st.columns([2, 3])
    with c1:
        st.markdown(f"**Name:** {selected['name']}")
        st.markdown(f"**Profession:** {selected['profession'] or '—'}")
        st.markdown(f"**Years of Experience:** {selected['years_experience'] if selected['years_experience'] is not None else '—'}")
        st.markdown(f"**Added:** {selected['created_at']}")
        st.markdown(f"**Notes:** {selected['notes'] or '—'}")

        # Editable extracted text
        raw_text = read_text_cache(selected["id"]) or "(no cached text)"
        with st.expander("Edit extracted text & Reindex"):
            new_text = st.text_area("Extracted CV text", value=raw_text, height=300)
            if st.button("🔁 Reindex from edited text"):
                if new_text.strip():
                    delete_candidate_vectors(selected["id"])  # purge
                    write_text_cache(selected["id"], new_text)
                    ingest_candidate_text(selected["id"], new_text, {
                        "name": selected["name"],
                        "profession": selected["profession"],
                        "years_experience": selected["years_experience"],
                    })
                    st.success("Reindexed from edited text.")
                else:
                    st.warning("Text is empty; nothing indexed.")

    with c2:
        st.markdown("**AI Summary**")
        if st.button("✨ Generate / Refresh Summary"):
            with st.spinner("Summarizing…"):
                summary = summarize_candidate(selected["id"], selected["name"])
                st.session_state["last_summary"] = summary
        st.write(st.session_state.get("last_summary", "Click the button to generate a summary."))


def render_chat_tab():
    st.subheader("💬 Chat (candidate‑grounded)")
    candidates = load_candidates()
    if not candidates:
        st.info("No candidates. Go to Manage CVs to add some.")
        return

    # Pick candidate for chat
    options = {f"{c['name']} — {c['profession'] or 'Unknown'} ({c['id'][:8]}…)": c for c in candidates}
    selected_label = st.selectbox("Select candidate for chat", list(options.keys()), key="chat_select")
    selected = options[selected_label]

    # Non-persistent chat state (session only)
    sid = f"chat_{selected['id']}"
    if sid not in st.session_state:
        st.session_state[sid] = []  # list of (role, content)

    cols = st.columns([1, 1])
    with cols[0]:
        if st.button("🧹 Clear chat"):
            st.session_state[sid] = []
            st.info("Chat cleared (not saved).")
    with cols[1]:
        st.caption("Chat is ephemeral and not stored to disk.")

    # Display history
    for role, msg in st.session_state[sid]:
        if role == "user":
            st.markdown(f"**You:** {msg}")
        else:
            st.markdown(f"**Assistant:** {msg}")

    # Input
    q = st.text_input("Ask about this candidate's experience, skills, projects, etc.", key="chat_input")
    if st.button("Ask") and q.strip():
        log(f"User question for candidate {selected['id']}: {q}")
        st.session_state[sid].append(("user", q))
        with st.spinner("Thinking…"):
            ans, sources = rag_answer(selected["id"], q)
        st.session_state[sid].append(("assistant", ans))
        # Show sources
        with st.expander("Sources"):
            if not sources:
                st.write("(no source previews)")
            else:
                for i, s in enumerate(sources, 1):
                    st.markdown(f"**{i}. score:** {s['score']}")
                    md = s.get("metadata", {})
                    st.json(md)
                    st.code(s.get("text_preview", ""))
        st.rerun()


# ---------- Main ----------

def main():
    ui_header()
    init_db()

    # Tabs
    t1, t2, t3, t4 = st.tabs(["📂 Manage CVs", "👤 Candidate", "💬 Chat", "🤖 Agent"])
    with t1:
        render_manage_tab()
    with t2:
        st.session_state["active_tab"] = "candidate"
        render_candidate_tab()
    with t3:
        render_chat_tab()
    with t4:
        render_ReAct_tab()


if __name__ == "__main__":
    main()
