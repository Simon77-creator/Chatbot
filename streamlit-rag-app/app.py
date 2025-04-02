import os
import io
import fitz
import pdfplumber
import tiktoken
import json
import streamlit as st
from typing import List, Dict
from collections import defaultdict
from azure.storage.blob import BlobServiceClient, ContentSettings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from openai import OpenAI

import streamlit as st

# Fallback für lokale Entwicklung
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")
QDRANT_API_KEY = st.secrets.get("QDRANT_API_KEY", "")
QDRANT_URL = st.secrets.get("QDRANT_URL", "")
AZURE_BLOB_CONN_STR = st.secrets.get("AZURE_BLOB_CONN_STR", "")
AZURE_CONTAINER = st.secrets.get("AZURE_CONTAINER", "")

# Optional: Warnung bei leerem Key (nur lokal sichtbar)
if not OPENAI_API_KEY:
    st.warning("⚠️ OPENAI_API_KEY nicht gefunden – läuft nur in Streamlit Cloud?")


# Initialisierung
openai_client = OpenAI(api_key=OPENAI_API_KEY)
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
blob_service = BlobServiceClient.from_connection_string(AZURE_BLOB_CONN_STR)
COLLECTION_NAME = "studienbot"
MEMORY_PREFIX = "memory"

# Erstelle Sammlung wenn nötig
if COLLECTION_NAME not in [c.name for c in qdrant.get_collections().collections]:
    qdrant.create_collection(COLLECTION_NAME, vectors_config=VectorParams(size=1536, distance=Distance.COSINE))

# PDF Verarbeitung
class PDFProcessor:
    def extract_text_chunks(self, pdf_bytes: bytes, max_tokens=800, overlap_tokens=60) -> List[Dict]:
        enc = tiktoken.encoding_for_model("text-embedding-ada-002")
        chunks = []

        def chunk_text(text):
            paragraphs = text.split("\n\n")
            token_buffer, current_tokens, result = [], 0, []
            for para in paragraphs:
                tokens = enc.encode(para)
                if current_tokens + len(tokens) > max_tokens:
                    result.append(enc.decode(token_buffer))
                    token_buffer = token_buffer[-overlap_tokens:] + tokens
                    current_tokens = len(token_buffer)
                else:
                    token_buffer += tokens
                    current_tokens += len(tokens)
            if token_buffer:
                result.append(enc.decode(token_buffer))
            return result

        with fitz.open(stream=io.BytesIO(pdf_bytes), filetype="pdf") as doc:
            for i, page in enumerate(doc):
                text = page.get_text()
                meta = {"source": "blob", "page": i + 1}
                for chunk in chunk_text(text):
                    chunks.append({"content": chunk, "metadata": meta})

        with pdfplumber.open(io.BytesIO(pdf_bytes)) as doc:
            for i, page in enumerate(doc.pages):
                tables = page.extract_tables()
                for table in tables:
                    table_text = "\n".join([" | ".join([str(cell) if cell else "" for cell in row]) for row in table if row])
                    meta = {"source": "blob", "page": i + 1}
                    for chunk in chunk_text(table_text):
                        chunks.append({"content": chunk, "metadata": meta})

        return chunks

# Vektor-Datenbank
class VectorDB:
    def __init__(self):
        self.client = qdrant
        self.collection = COLLECTION_NAME

    def add(self, documents: List[Dict]):
        points = []
        for i, doc in enumerate(documents):
            emb = openai_client.embeddings.create(input=doc["content"], model="text-embedding-ada-002")
            vec = emb.data[0].embedding
            points.append(PointStruct(id=i, vector=vec, payload={"text": doc["content"], **doc["metadata"]}))
        self.client.upsert(collection_name=self.collection, points=points)

    def query(self, question: str, n=30) -> List[Dict]:
        emb = openai_client.embeddings.create(input=question, model="text-embedding-ada-002")
        query_vector = emb.data[0].embedding
        results = self.client.search(collection_name=self.collection, query_vector=query_vector, limit=n)
        return [{
            "text": r.payload["text"],
            "source": r.payload["source"],
            "page": r.payload["page"],
            "score": r.score
        } for r in results]

# Memory Funktionen
def list_sessions(user: str) -> List[str]:
    prefix = f"{MEMORY_PREFIX}/{user}/"
    container = blob_service.get_container_client(AZURE_CONTAINER)
    return [b.name.replace(prefix, "").replace(".json", "") for b in container.list_blobs(name_starts_with=prefix)]

def load_memory(user: str, session: str) -> List[Dict]:
    try:
        path = f"{MEMORY_PREFIX}/{user}/{session}.json"
        blob = blob_service.get_blob_client(container=AZURE_CONTAINER, blob=path)
        return json.loads(blob.download_blob().readall().decode("utf-8"))
    except:
        return []

def save_memory(user: str, session: str, history: List[Dict]):
    path = f"{MEMORY_PREFIX}/{user}/{session}.json"
    blob = blob_service.get_blob_client(container=AZURE_CONTAINER, blob=path)
    blob.upload_blob(json.dumps(history), overwrite=True, content_settings=ContentSettings(content_type="application/json"))

def delete_memory(user: str, session: str):
    path = f"{MEMORY_PREFIX}/{user}/{session}.json"
    blob = blob_service.get_blob_client(container=AZURE_CONTAINER, blob=path)
    blob.delete_blob()

# Kontextaufbereitung
def prepare_context_chunks(resultate, max_tokens=6500, max_chunk_length=2000, max_per_source=6):
    enc = tiktoken.encoding_for_model("gpt-4")
    total_tokens = 0
    context_chunks = []
    seen = set()
    source_counter = defaultdict(int)

    # Optional: zuerst nur relevante Scores (z. B. unter 0.82)
    resultate = [r for r in resultate if r["score"] <= 0.82]

    # Nach Score sortieren (beste zuerst)
    resultate = sorted(resultate, key=lambda x: x["score"])

    for r in resultate:
        if source_counter[r["source"]] >= max_per_source:
            continue

        text = r["text"][:max_chunk_length].strip()
        tokens = len(enc.encode(text))

        if total_tokens + tokens > max_tokens:
            break

        if text.lower() in seen:
            continue

        seen.add(text.lower())
        context_chunks.append({
            "text": text,
            "source": r["source"],
            "page": r["page"]
        })

        source_counter[r["source"]] += 1
        total_tokens += tokens

    return context_chunks


# Prompt-Aufbau
def build_gpt_prompt(context_chunks: List[Dict], frage: str) -> List[Dict]:
    context = "\n\n".join([f"{doc['source']} – Seite {doc['page']}:\n{doc['text']}" for doc in context_chunks])
    system_prompt = (
    "Du bist ein präziser Studienberater der FHDW.\n"
    "Antworte sachlich, vollständig und strukturiert auf Grundlage des folgenden Kontexts.\n"
    "Beziehe dich klar auf die gefundenen Informationen – zitiere ggf. Seitenangaben.\n"
    "Wenn der Kontext keine Antwort erlaubt, weise freundlich darauf hin.\n"
    "Vermeide allgemeine Aussagen, wenn sie nicht im Kontext vorkommen.\n\n"
    f"{context}"
)

    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": frage}
    ]

# Hauptabfrage
def run_query(frage, user, session, memory_on=True, debug=False):
    db = VectorDB()
    memory = load_memory(user, session) if memory_on else []
    resultate = db.query(frage)
    chunks = prepare_context_chunks(resultate)
    messages = memory + build_gpt_prompt(chunks, frage)
    response = openai_client.chat.completions.create(model="gpt-4", messages=messages, max_tokens=1500, temperature=0.4)
    reply = response.choices[0].message.content
    if memory_on:
        memory.extend([{"role": "user", "content": frage}, {"role": "assistant", "content": reply}])
        save_memory(user, session, memory)
    if debug:
        st.subheader("🧪 DEBUG INFO")
        st.write("Suchergebnisse aus Vektor-DB:")
        st.write(resultate)
        st.write("Generierter Kontext für Prompt:")
        st.write(chunks)
        st.write("Nachricht an GPT:")
        st.write(messages)
    return reply

# Streamlit UI
st.title("Studienbot")
st.sidebar.header("Einstellungen")

user = st.sidebar.text_input("Benutzername", value="demo")
sessions = list_sessions(user)
selected_session = st.sidebar.selectbox("Sitzung wählen oder neu", ["Neue Sitzung"] + sessions)
session = st.sidebar.text_input("Neue Sitzung" if selected_session == "Neue Sitzung" else "Sitzung", value="default" if selected_session == "Neue Sitzung" else selected_session)

if st.sidebar.button("Verlauf löschen"):
    delete_memory(user, session)
    st.success("Verlauf gelöscht.")

use_memory = st.sidebar.checkbox("Memory verwenden", value=True)
frage = st.text_input("Stelle deine Frage:")

if st.button("Absenden") and frage:
    antwort = run_query(frage, user, session, use_memory, debug=True)
    st.markdown(f"**Antwort:**\n\n{antwort}")
    if use_memory:
        st.markdown("## Chatverlauf")
        for entry in load_memory(user, session):
            role = "Du" if entry["role"] == "user" else "Bot"
            st.markdown(f"**{role}:** {entry['content']}")
