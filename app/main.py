from pathlib import Path

from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel

from app.pdf_loader import extract_text_from_pdf
from app.text_chunker import chunk_text
from app.embeddings import EmbeddingModel
from app.vector_store import VectorStore
from app.retriever import Retriever
from app.rag_pipeline import RAGPipeline


app = FastAPI(title="AI Research Assistant")


class QueryRequest(BaseModel):
    question: str


UPLOAD_DIR = Path("data/uploads")
INDEX_DIR = Path("data/index")

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)

INDEX_PATH = INDEX_DIR / "faiss.index"
METADATA_PATH = INDEX_DIR / "metadata.pkl"

embedder = EmbeddingModel()
store = None
retriever = None
rag = None


def build_pipeline_from_pdf(pdf_path: str, source_file: str):
    global store, retriever, rag

    pages = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(pages, source_file)

    texts = [chunk["text"] for chunk in chunks]
    embeddings = embedder.encode_texts(texts)

    dim = embeddings.shape[1]
    store = VectorStore(dim)
    store.add(embeddings, chunks)

    store.save(str(INDEX_PATH), str(METADATA_PATH))

    retriever = Retriever(embedder, store)
    rag = RAGPipeline(retriever)


def load_existing_pipeline():
    global store, retriever, rag

    if INDEX_PATH.exists() and METADATA_PATH.exists():
        store = VectorStore(dim=384)
        store.load(str(INDEX_PATH), str(METADATA_PATH))
        retriever = Retriever(embedder, store)
        rag = RAGPipeline(retriever)
        return True

    return False


# Try loading saved index first
loaded = load_existing_pipeline()

# If nothing saved, try loading sample PDF
if not loaded:
    default_pdf = UPLOAD_DIR / "sample.pdf"
    if default_pdf.exists():
        build_pipeline_from_pdf(str(default_pdf), default_pdf.name)


@app.get("/")
def root():
    return {"message": "AI Research Assistant API is running"}


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        return {"error": "Only PDF files are allowed."}

    save_path = UPLOAD_DIR / file.filename

    with open(save_path, "wb") as f:
        content = await file.read()
        f.write(content)

    build_pipeline_from_pdf(str(save_path), file.filename)

    return {
        "message": "PDF uploaded and indexed successfully.",
        "filename": file.filename,
        "index_saved": True
    }


@app.post("/ask")
def ask_question(request: QueryRequest):
    if rag is None:
        return {"error": "No PDF has been indexed yet. Please upload a PDF first."}

    response = rag.answer(request.question, top_k=3)
    return response