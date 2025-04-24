from langchain.chains import RetrievalQA,create_retrieval_chain
from .prompt_template import get_prompt_template, get_qa_system_prompt
from langchain.chains.combine_documents import create_stuff_documents_chain
from .llm_instance import llm
from .vectorestore import get_vectorstore
from .retriever import get_history_aware_retriever, get_similarity_search_retriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from .chat_session_manager import get_session_history
from langchain_core.runnables.history import RunnableWithMessageHistory

#Retrieval QA chain stuff type, others are Map Reduce/Refine etc. 
def get_retrieval_qa_chain(chain_type="stuff"):
    retriever = get_vectorstore()
    prompt = get_prompt_template()

    qa_chain = RetrievalQA.from_chain_type(
        llm = llm,
        retriever = retriever.as_retriever(),
        chain_type = chain_type,
        return_source_documents=True,
        chain_type_kwargs = {"prompt": prompt}
    )

    return qa_chain

def get_rag_retrieval_chain(retriever, format_docs, prompt, llm):

    rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
    )

    return rag_chain



#for convesational rag chain with chat history with LCEL

def get_question_answer_chain():
    qa_prompt = get_qa_system_prompt()

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    return question_answer_chain

def get_rag_chain():
    qa_chain = get_question_answer_chain()
    history_aware_retriever = get_history_aware_retriever()

    rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)
    return rag_chain

def get_conversational_rag_chain():
    rag_chain = get_rag_chain()

    conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
    )

    return conversational_rag_chain