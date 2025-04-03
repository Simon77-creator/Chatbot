import streamlit as st
import os, fitz, openai, pdfplumber
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
from typing import List, Dict
from collections import defaultdict
import tiktoken

# Set secrets
try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    AZURE_BLOB_CONN_STR = st.secrets["AZURE_BLOB_CONN_STR"]
    AZURE_CONTAINER = st.secrets["AZURE_CONTAINER"]
    QDRANT_HOST = st.secrets["QDRANT_HOST"]
except KeyError as e:
    st.error(f"Fehlender Schlüssel in den Geheimnissen: {e}")
    st.stop()

client = openai.OpenAI(api_key=OPENAI_API_KEY)

# PDF Processor
class PDFProcessor:
    def extract_text_chunks(self, pdf_path: str, max_tokens=2000, overlap_tokens=50) -> List[Dict]:
        chunks = []
        enc = tiktoken.encoding_for_model("gpt-4")

        def paragraph_chunks(text: str) -> List[str]:
            paragraphs = text.split("\n\n")
            token_buffer = []
            current_tokens = 0
            result = []

            for para in paragraphs:
                para_tokens = enc.encode(para)
                if current_tokens + len(para_tokens) > max_tokens:
                    result.append(enc.decode(token_buffer))
                    token_buffer = token_buffer[-overlap_tokens:] + para_tokens
                    current_tokens = len(token_buffer)
                else:
                    token_buffer += para_tokens
                    current_tokens += len(para_tokens)

            if token_buffer:
                result.append(enc.decode(token_buffer))
            return result

        try:
            with fitz.open(pdf_path) as doc:
                for page_num, page in enumerate(doc):
                    text = page.get_text()
                    metadata = {"source": os.path.basename(pdf_path), "page": page_num + 1}
                    for chunk in paragraph_chunks(text):
                        chunks.append({"content": chunk, "metadata": metadata})
        except Exception as e:
            st.error(f"Fehler bei Text in {pdf_path}: {e}")

        try:
            with pdfplumber.open(pdf_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    tables = page.extract_tables()
                    for table in tables:
                        table_text = "\n".join([
                            " | ".join([str(cell) if cell is not None else "" for cell in row])
                            for row in table if row
                        ])
                        metadata = {"source": os.path.basename(pdf_path), "page": i + 1}
                        for chunk in paragraph_chunks(table_text):
                            chunks.append({"content": chunk, "metadata": metadata})
        except Exception as e:
            st.error(f"Fehler bei Tabellen in {pdf_path}: {e}")

        return chunks

# Qdrant Vector Database
class VectorDB:
    def __init__(self, api_key: str, host: str, port=6333):
        try:
            self.client = QdrantClient(api_key=api_key, host=host, port=port)
            self.collection_name = "studienbot"
            self.client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "size": 1536,
                    "distance": "Cosine"
                }
            )
        except ValueError as e:
            st.error(f"Fehler bei der Verbindung zu Qdrant: {e}")
            st.stop()

    def add(self, documents: List[Dict]):
        points = [
            PointStruct(id=f"doc_{i}", vector=self.client.get_embeddings(doc["content"]), payload=doc["metadata"])
            for i, doc in enumerate(documents)
        ]
        self.client.upsert(collection_name=self.collection_name, points=points)

    def query(self, question: str, n=30) -> List[Dict]:
        embeddings = self.client.get_embeddings(question)
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=embeddings,
            limit=n
        )
        hits = []
        for result in search_result:
            hits.append({
                "text": result.payload["content"],
                "source": result.payload["source"],
                "page": result.payload["page"],
                "score": result.score
            })
        return hits

# Relevance Sorting and Summarization
def prepare_context_chunks(resultate: List[Dict], max_tokens=6500, max_chunk_length=2000, max_per_source=4, allow_duplicates: bool = False):
    seen_texts = set()
    if resultate and "score" in resultate[0]:
        resultate = sorted(resultate, key=lambda x: x["score"])

    def summarize_chunk(text: str) -> str:
        if len(text.split()) < 30:
            return text
        prompt = f"Fasse folgenden Abschnitt kurz und sachlich zusammen:\n\n{text}"
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Du bist ein sachlicher Assistent."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=300
        )
        return response.choices[0].message.content.strip()

    enc = tiktoken.encoding_for_model("gpt-4")
    total_tokens = 0
    context_chunks = []
    source_counter = defaultdict(int)

    for r in resultate:
        source = r["source"]
        if source_counter[source] >= max_per_source:
            continue

        text = r["text"][:max_chunk_length].strip()
        if len(enc.encode(text)) > max_chunk_length:
            text = summarize_chunk(text)

        if len(text) < 50:
            continue

        norm_text = text.lower().strip()
        if not allow_duplicates and norm_text in seen_texts:
            continue
        seen_texts.add(norm_text)

        tokens = len(enc.encode(text))
        if total_tokens + tokens > max_tokens:
            break

        context_chunks.append({"text": text, "source": source, "page": r["page"]})
        total_tokens += tokens
        source_counter[source] += 1

    return context_chunks

# Prompt Construction
def build_gpt_prompt(context_chunks: List[Dict], frage: str) -> List[Dict]:
    context = "\n\n".join([
        f"### {doc['source']} – Seite {doc['page']}\n{doc['text']}" for doc in context_chunks
    ])

    system_prompt = (
        "Du bist ein freundlicher und präziser Studienberater der FHDW.\n"
        "Nutze ausschließlich den folgenden Kontext, um die Nutzerfrage zu beantworten.\n"
        "Wenn relevante Informationen enthalten sind, fasse sie vollständig, korrekt und strukturiert zusammen.\n"
        "Wenn keine passende Information im Kontext vorhanden ist, sage das ehrlich.\n"
        "Strukturiere deine Antwort klar: Absätze, Aufzählungen, ggf. Zwischenüberschriften.\n"
        "Zitiere wichtige Begriffe oder Formulierungen wörtlich, wenn möglich.\n"
        "Verwende Emojis nur, wenn es zur besseren Lesbarkeit beiträgt.\n\n"
        f"### Kontext ###\n{context}\n\n### Aufgabe ###\nFrage: {frage}"
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": frage}
    ]

# Streamlit UI
st.title("Studienberater der FHDW")

blob_service_client = BlobServiceClient.from_connection_string(AZURE_BLOB_CONN_STR)
container_client = blob_service_client.get_container_client(AZURE_CONTAINER)

blobs = container_client.list_blobs()
pdf_files = [blob.name for blob in blobs if blob.name.endswith('.pdf')]

if pdf_files:
    processor = PDFProcessor()
    db = VectorDB(OPENAI_API_KEY, QDRANT_HOST)

    all_chunks = []
    for pdf_file in pdf_files:
        blob_client = container_client.get_blob_client(pdf_file)
        pdf_path = os.path.join("/tmp", pdf_file)
        with open(pdf_path, "wb") as f:
            f.write(blob_client.download_blob().readall())
        chunks = processor.extract_text_chunks(pdf_path)
        all_chunks.extend(chunks)
        os.remove(pdf_path)  # Entferne temporäre PDF-Datei

    st.write(f"Speichere {len(all_chunks)} Chunks in der Datenbank...")
    db.add(all_chunks)
    st.write("Verarbeitung abgeschlossen.")

frage = st.text_input("Stelle eine Frage:")

if frage:
    db = VectorDB(OPENAI_API_KEY, QDRANT_HOST)
    resultate = db.query(frage, n=30)
    kontext = prepare_context_chunks(resultate)

    if not kontext:
        st.write("Ich konnte zu deiner Frage in den hinterlegten Dokumenten leider keine passenden Informationen finden.")
    else:
        st.write("Kontext, den GPT erhalten hat:")
        for chunk in kontext:
            st.write(f"🔹 {chunk['source']} – Seite {chunk['page']}\n{chunk['text']}\n{'-'*60}")

        messages = build_gpt_prompt(kontext, frage)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.3,
            max_tokens=1500
        )

        antwort = response.choices[0].message.content
        st.markdown(antwort)