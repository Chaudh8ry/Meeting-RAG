from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


# ---------------- CONFIGURATION ----------------

# Where Chroma will store the persistent vector database
CHROMA_DIR = "vector_db"

# Name of the Chroma collection
COLLECTION_NAME = "meeting_transcript"

# Embedding model used to convert text into vectors
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# ---------------- EMBEDDINGS ----------------

def get_embeddings():
    # Create the embedding model.
    # It converts text into numerical vectors for semantic search.
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        
        # Run the embedding model on CPU
        model_kwargs={"device": "cpu"}
    )


# ---------------- BUILD VECTOR STORE ----------------

def build_vector_store(transcript: str) -> Chroma:
    print("Building vector store")

    # Split the large transcript into smaller overlapping chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,       # Approximate size of each chunk in characters
        chunk_overlap=50      # Keep some context between consecutive chunks
    )

    # Convert transcript → list of text chunks
    chunks = splitter.split_text(transcript)

    # Convert each chunk into a LangChain Document
    # Document contains the actual text + metadata
    docs = [
        Document(
            page_content=chunk,
            metadata={"chunk_index": i}
        )
        for i, chunk in enumerate(chunks)
    ]

    # Get the embedding model
    embeddings = get_embeddings()

    # Create Chroma vector store from the documents
    # Chroma generates embeddings and stores them persistently
    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR
    )

    return vector_store


# ---------------- LOAD EXISTING VECTOR STORE ----------------

def load_vector_store() -> Chroma:
    # We need the same embedding model for query embeddings
    embeddings = get_embeddings()

    # Connect to the already-existing Chroma database
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR
    )

    return vector_store


# ---------------- CREATE RETRIEVER ----------------

def get_retriever(vector_store: Chroma, k: int = 4):
    # Convert the vector store into a retriever.
    # It will return the top-k most similar chunks for a query.
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )
