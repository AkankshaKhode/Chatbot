import os
import sqlite3
from typing import List
import PyPDF2
import streamlit as st
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import torch


# 1. Content Extraction
def extract_text_from_pdf(pdf_path: str) -> List[str]:
    """Extract text from a PDF file."""
    pdf_reader = PyPDF2.PdfReader(pdf_path)
    extracted_text = []
    for page in pdf_reader.pages:
        extracted_text.append(page.extract_text())
    return extracted_text

# 2. Hierarchical Indexing
def create_hierarchical_index(book_text: List[str], db_path: str):
    """Create a hierarchical index and store it in SQLite."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS index_tree (
            id INTEGER PRIMARY KEY,
            parent_id INTEGER,
            level TEXT,
            content TEXT
        )
    """)
    parent_id = None
    for i, chapter in enumerate(book_text):
        cursor.execute("INSERT INTO index_tree (parent_id, level, content) VALUES (?, ?, ?)", (parent_id, f"Chapter {i+1}", chapter))
        parent_id = cursor.lastrowid
    conn.commit()
    conn.close()

# 3. Retrieval Techniques
def search_content(query: str, db_path: str, model) -> List[str]:
    """Retrieve relevant content using semantic search."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT content FROM index_tree")
    all_content = [row[0] for row in cursor.fetchall()]

    if not all_content:
        return []

    # Semantic Search
    query_embedding = model.encode(query, convert_to_tensor=True)
    corpus_embeddings = model.encode(all_content, convert_to_tensor=True)
    semantic_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)

    top_k = min(5, len(all_content))  # Ensure k does not exceed the number of content items
    top_indices = torch.topk(semantic_scores[0], top_k).indices.tolist()

    conn.close()
    return [all_content[i] for i in top_indices]

# 4. RAG System
def generate_answer(contexts: List[str], query: str) -> str:
    """Generate answers using a Retrieval-Augmented Generation model."""
    qa_pipeline = pipeline("question-answering")
    answers = []
    for context in contexts:
        result = qa_pipeline(question=query, context=context)
        answers.append((result['answer'], result['score']))
    return max(answers, key=lambda x: x[1])[0]

# 5. Streamlit User Interface
def main():
    st.title("LLM-Powered Question Answering System")

    # Paths to multiple PDF files
    pdf_paths = ["book1.pdf", "book2.pdf", "book3.pdf"]
    all_text = []

    # Extract and Index Text from multiple PDFs
    for pdf_path in pdf_paths:
        if os.path.exists(pdf_path):
            with st.spinner(f"Extracting text from {pdf_path}..."):
                text = extract_text_from_pdf(pdf_path)
                all_text.extend(text)

    # Save all extracted text to SQLite
    if all_text:
        db_path = "hierarchical_index.db"
        create_hierarchical_index(all_text, db_path)

    # Query Input
    query = st.text_input("Enter your query:")
    if query:
        model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

        with st.spinner("Searching..."):
            contexts = search_content(query, db_path, model)
            if not contexts:
                st.warning("No relevant content found.")
                return
        with st.spinner("Generating answer..."):
            answer = generate_answer(contexts, query)

        st.write("**Answer:**", answer)

if __name__ == "__main__":
    main()
