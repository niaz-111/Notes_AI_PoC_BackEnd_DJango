from langchain.schema.document import Document
# from .document_loader import load_notes
from .text_splitter import split_text
from .vectorestore import add_documents,add_documents_with_ids
from .text_splitter import chunk_note

def run_ingestion_pipeline(notes):
    # notes = load_notes()
    all_documents = []
    all_ids = []

    for note in notes:
        docs, ids = chunk_note(note["uid"], note["title"], note["content"])
        print("docs: ", docs)
        print("ids : ", ids)
        all_documents.extend(docs)
        all_ids.extend(ids)

    add_documents_with_ids(docs=all_documents, ids=all_ids)
    print(f"✅ Ingested {len(all_documents)} chunks into ChromaDB.")