# chatbot/langchain_pipeline/retriever.py
from .vectorestore import get_vectorstore
from .llm_instance import llm
from langchain.retrievers import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from .prompt_template import get_contexualize_history_aware_q_prompt, get_contexualize_history_aware_q_prompt_image_caption


def get_similarity_search_retriever(k: int = 3, fetch_k: int = 20, metadata_filter=None):
    vectordb = get_vectorstore()

    search_kwargs = {"k": k}
    if metadata_filter is not None:
        search_kwargs["filter"] = metadata_filter

    retriever = vectordb.as_retriever(search_kwargs=search_kwargs)
    return retriever



def get_history_aware_retriever():
    #contextualize_q_prompt = get_contexualize_history_aware_q_prompt()
    contextualize_q_prompt = get_contexualize_history_aware_q_prompt_image_caption()
    #retiriever = get_similarity_search_retriever()
    #retiriever = get_selfquery_retriever()
    retiriever = get_mmr_retriever()

    history_aware_retriever = create_history_aware_retriever(
        llm, retiriever, contextualize_q_prompt
    )

    return history_aware_retriever


def retrieve_similar_documents(query: str):
    retriever = get_similarity_search_retriever()
    return retriever.get_relevant_documents(query)

def retrieve_similar_documents_with_scores(query: str, k: int = 3):
    vectordb = get_vectorstore()
    results = vectordb.similarity_search_with_score(query, k=k)

    # results is a list of tuples: (Document, score)
    return results


def get_mmr_retriever(k: int = 50, fetch_k: int = 150, lambda_mult: float = 0.5):
    vectordb = get_vectorstore()
    retriever = vectordb.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": k,                
            "fetch_k": fetch_k,      
            "lambda_mult": lambda_mult
        }
    )

    return retriever


def get_selfquery_retriever(k: int = 50, fetch_k: int = 150, metadata_filter=None):
    vectordb = get_vectorstore()

    metadata_field_info = [
        AttributeInfo(
            name="uid",
            description="The unique id for identifying the document",
            type="string",
        ),
        AttributeInfo(
            name="title",
            description="The title of the document",
            type="string",
        ),
        AttributeInfo(
        name="contains_image",
        description="True if the chunk includes an image caption or path",
        type="bool"
        ),
    ]

    document_contents = "A chunk of a document containing relevant information for answering user queries."

    search_kwargs = {"k": k, "fetch_k": fetch_k}
    if metadata_filter:
        search_kwargs["filter"] = metadata_filter

    retriever = SelfQueryRetriever.from_llm(
        llm=llm,
        vectorstore=vectordb,
        document_contents = document_contents,
        document_content_description = document_contents,
        metadata_field_info=metadata_field_info,
        search_kwargs={"k": k},
    )

    return retriever


def retrieve_compressed_documents(query: str, k: int = 3):
    vectordb = get_vectorstore()

    base_retriever = vectordb.as_retriever(search_kwargs={"k": k})
    compressor = LLMChainExtractor.from_llm(llm)

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    )

    return compression_retriever.get_relevant_documents(query)

