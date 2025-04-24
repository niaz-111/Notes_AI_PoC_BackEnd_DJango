# chatbot/langchain_pipeline/text_splitter.py
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document

def split_text(text: str, chunk_size=500, chunk_overlap=50) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)


def chunk_note(note_uid: str, title: str, content: str) -> tuple[list[Document], list[str]]:
    chunks = split_text(content)
    documents = []
    ids = []

    for i, chunk in enumerate(chunks):
        doc = Document(
            page_content = f"[Title: {title}] [UID: {note_uid}]\n{chunk}",
            metadata={"uid": note_uid, "title": title, "chunk_id": i}
        )
        documents.append(doc)
        ids.append(f"{note_uid}_chunk_{i}")
    
    return documents, ids

