# chatbot/langchain_pipeline/vectorstore.py
from langchain.vectorstores import Chroma
from langchain.schema.document import Document
from .llm_instance import embedding_model

def get_vectorstore(persist_dir="chroma_store", collection_name="notes_collection"):
    return Chroma(embedding_function=embedding_model, collection_name=collection_name, persist_directory=persist_dir)

def add_documents(docs: list[Document], collection_name="notes_collection", persist_dir="chroma_store"):
    vectordb = Chroma.from_documents(docs, embedding_model, persist_directory=persist_dir, collection_name=collection_name)
    vectordb.persist()
    return vectordb

def add_text_spilts(splits : list[str], persist_dir="chroma_store"):
    vectordb = Chroma.from_documents(splits, embedding_model, persist_directory=persist_dir)
    vectordb.persist()
    return vectordb

def cleanup_vectorstore(collection_name="notes_collection"):
    vectordb = get_vectorstore(collection_name=collection_name)
    vectordb.delete_collection()


def add_documents_with_ids(docs: list[Document], ids: list[str] = None, collection_name="notes_collection", persist_dir="chroma_store"):

    vectordb = Chroma(
        collection_name=collection_name,
        embedding_function=embedding_model,
        persist_directory=persist_dir
    )

    vectordb.add_documents(documents=docs, ids=ids)
    vectordb.persist()

    return vectordb



def delete_note_by_uid(note_uid: str):
    vectorstore = get_vectorstore()
    collection = vectorstore._collection
    all_ids = collection.get()["ids"] 

    matching_ids = [doc_id for doc_id in all_ids if doc_id.startswith(f"{note_uid}_chunk_")]

    if not matching_ids:
        print(f"No chunks found for UID '{note_uid}'")
        return

    vectorstore.delete(ids=matching_ids)
    vectorstore.persist()
    print(f"Deleted {len(matching_ids)} chunks for UID '{note_uid}'")




