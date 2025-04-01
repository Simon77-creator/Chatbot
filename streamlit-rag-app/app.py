import streamlit as st
import fitz
import pdfplumber
import io
import openai
import tiktoken
import json
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from azure.storage.blob import BlobServiceClient, ContentSettings
from typing import List, Dict

# ======= STREAMLIT CONFIG =======
st.set_page_config(page_title="📚 Studienbot", layout="wide")
st.title("📚 Studienbot mit GPT + Qdrant + Azure")

# ======= SECRETS =======
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
QDRANT_API_KEY = st.secrets["QDRANT_API_KEY"]
QDRANT_URL = st.secrets["QDRANT_URL"]
AZURE_BLOB_CONN_STR = st.secrets["AZURE_BLOB_CONN_STR"]
AZURE_CONTAINER = st.secrets["AZURE_CONTAINER"]

openai.api_key = OPENAI_API_KEY
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
blob_service = BlobServiceClient.from_connection_string(AZURE_BLOB_CONN_STR)

COLLECTION_NAME = "studienbot"
MEMORY_PREFIX = "memory"

if COLLECTION_NAME not in [c.name for c in qdrant.get_collections().collections]:
    qdrant.recreate_collection(COLLECTION_NAME, vectors_config=VectorParams(size=1536, distance=Distance.COSINE))

# ======= UI: User und Session =======
with st.sidebar:
    st.header("🧠 Sitzungs-Memory")
    user = st.text_input("👤 Dein Name", value="gast")
    session = st.text_input("📁 Sitzungstitel", value="standard")
    if st.button("🔁 PDFs neu verarbeiten"):
        process_pdfs_from_blob()

memory_blob_path = f"{MEMORY_PREFIX}/{user}/{session}.json"

# ======= SPEICHERN & LADEN =======
def load_memory() -> List[Dict]:
    try:
        blob_client = blob_service.get_blob_client(AZURE_CONTAINER, memory_blob_path)
        data = blob_client.download_blob().readall()
        return json.loads(data.decode("utf-8"))
    except:
        return []

def save_memory(history: List[Dict]):
    blob_client = blob_service.get_blob_client(AZURE_CONTAINER, memory_blob_path)
    blob_client.upload_blob(json.dumps(history), overwrite=True, content_settings=ContentSettings(content_type="application/json"))

# ======= PDF CHUNKING =======
def extract_chunks_from_pdf(pdf_bytes: bytes, max_tokens=800, overlap=60) -> List[Dict]:
    enc = tiktoken.encoding_for_model("text-embedding-ada-002")
    chunks = []

    def chunk_text(text):
        paragraphs = text.split("\n\n")
        buffer, count, results = [], 0, []
        for para in paragraphs:
            tokens = enc.encode(para)
            if count + len(tokens) > max_tokens:
                results.append(enc.decode(buffer))
                buffer = buffer[-overlap:] + tokens
                count = len(buffer)
            else:
                buffer += tokens
                count += len(tokens)
        if buffer:
            results.append(enc.decode(buffer))
        return results

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

# ======= BLOB-PROZESSOR =======
def process_pdfs_from_blob():
    container_client = blob_service.get_container_client(AZURE_CONTAINER)
    for blob in container_client.list_blobs():
        if blob.name.endswith(".pdf"):
            st.write(f"📥 Verarbeite: {blob.name}")
            blob_data = container_client.get_blob_client(blob).download_blob().readall()
            chunks = extract_chunks_from_pdf(blob_data)
            vectors = []
            for chunk in chunks:
                emb = openai.Embedding.create(input=chunk["content"], model="text-embedding-ada-002")
                vectors.append(emb.data[0].embedding)
            points = [
                PointStruct(id=i, vector=vectors[i], payload={"text": chunks[i]["content"], **chunks[i]["metadata"]})
                for i in range(len(chunks))
            ]
            qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
    st.success("✅ Alle PDFs verarbeitet und in Qdrant gespeichert.")

# ======= QUERY MIT MEMORY =======
def run_query(frage):
    history = load_memory()

    emb = openai.Embedding.create(input=frage, model="text-embedding-ada-002")
    results = qdrant.search(collection_name=COLLECTION_NAME, query_vector=emb.data[0].embedding, limit=15)
    kontext = [f"[{r.payload['source']} – Seite {r.payload['page']}]:\n{r.payload['text']}" for r in results]
    context_text = "\n\n".join(kontext)

    messages = history + [
        {"role": "system", "content": "Du bist ein Studienberater der FHDW. Antworte strukturiert und ausführlich."},
        {"role": "user", "content": f"Kontext:\n{context_text}\n\nFrage: {frage}"}
    ]

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages,
        temperature=0.4,
        max_tokens=1200
    )

    reply = response.choices[0].message.content
    st.subheader("💬 Antwort")
    st.write(reply)

    history.append({"role": "user", "content": frage})
    history.append({"role": "assistant", "content": reply})
    save_memory(history)

# ======= UI =======
frage = st.text_input("Stelle deine Frage an den Studienbot:")
if frage:
    run_query(frage)
