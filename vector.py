from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd

# Load your Jarvis knowledge base
df = pd.read_csv("jarvis_knowledge.csv")

# Initialize embeddings model
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# Set up Chroma DB path
db_location = "./jarvis_vector_db"
add_documents = not os.path.exists(db_location)

# If DB doesn't exist, prepare documents
if add_documents:
    documents = []
    ids = []

    for i, row in df.iterrows():
        title = str(row.get("Title", ""))
        review = str(row.get("Review", ""))
        rating = str(row.get("Rating", ""))
        date = str(row.get("Date", ""))

        doc = Document(
            page_content=title + " " + review,
            metadata={"rating": rating, "date": date}
        )
        documents.append(doc)
        ids.append(str(i))

# Create or load Chroma vector DB
vector_store = Chroma(
    collection_name="jarvis_knowledge_base",
    persist_directory=db_location,
    embedding_function=embeddings
)

# If new, add documents to the vector store
if add_documents:
    vector_store.add_documents(documents=documents, ids=ids)

# Export retriever for use in main.py
retriever = vector_store.as_retriever(search_kwargs={"k": 5})
