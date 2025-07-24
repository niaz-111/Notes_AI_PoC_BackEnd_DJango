from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

def get_prompt_template():
    template = """Use the following pieces of context to answer the question at the end. \
    If you don't know the answer, just say that you don't know, don't try to make up an answer. \
    Use three sentences maximum. Keep the answer as concise as possible. \
    Always say "thanks for asking!" at the end of the answer. 
    {context}
    Question: {question}
    Helpful Answer:"""
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

    return QA_CHAIN_PROMPT


def get_prompt_template_image_caption():
    template = """Use the following pieces of context to answer the question at the end. \
The context may include image captions like `[Image Caption: ...]`, image paths like `[Image Path: ...]`, \
or handwritten content marked with `[HandWriting Content Start] ... [HandWriting Content End]`. \
Treat the content inside these tags as handwritten content, but only if there is meaningful text between them. \
If the tags are empty or contain only whitespace or newlines, assume no handwriting is available. \
If the answer involves an image or handwriting, refer to it directly. \
If you don't know the answer, say so — don't make anything up. \
Use three sentences maximum. Keep the answer as concise as possible. \
Always say "thanks for asking!" at the end of the answer.
{context}
Question: {question}
Helpful Answer:"""
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
    return QA_CHAIN_PROMPT



def get_contexualize_history_aware_q_prompt_image_caption():
    contextualize_q_system_prompt = """You are a helpful assistant tasked with reformulating questions in a conversation. \
Given the chat history and the latest user question — which may refer to information, images, or handwritten content mentioned earlier — \
rewrite the question into a standalone version that makes sense without prior context. \
Be especially careful to preserve references to any images, diagrams, or handwritten content \
(e.g., content between `[HandWriting Content Start]` and `[HandWriting Content End]`) if they are mentioned. \
Only preserve references to handwritten content if the content is meaningful (i.e., not empty or just whitespace or newlines). \
If the tags were present but contained no usable content, treat them as if no handwriting was provided. \
If the question already stands alone or is a simple greeting like \"hello\", \"hi\", or \"hey\", return it unchanged. \
Do NOT answer the question, only rewrite it."""

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    return contextualize_q_prompt




def get_contexualize_history_aware_q_prompt():
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    return contextualize_q_prompt

"""You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know. \
Use three sentences maximum and keep the answer concise.\
"""

def get_qa_system_prompt_image_caption():
    qa_system_prompt = """You will be given a chat history, a question, and relevant context documents.

Each document contains a `title`, `uid`, and text content — possibly including `[Image Caption: ...]`, `[Image Path: ...]`, or handwritten content marked with `[HandWriting Content Start] ... [HandWriting Content End]`.

Treat content inside the handwriting tags as handwritten content, but only if there is meaningful (non-empty, not just whitespace/newlines) text between the tags. If the tags are empty or contain no usable content, ignore them and treat the document as not containing handwriting.

Follow these rules:
- If the question is a greeting like "hello", "hi", or similar, respond with a friendly greeting — don't reference any context.
- If the question is not answerable from the context or chat history, say:
  "This question seems unrelated to the notes. However, based on my own knowledge, here’s what I can say:"
  Then generate your best answer.
- If the answer is not known, say so — do not fabricate it.
- Keep answers concise and to the point — ideally within three sentences.

{context}"""

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    return qa_prompt




def get_qa_system_prompt():
    qa_system_prompt = """You will be given a chat history, a question, and relevant context documents. \
Each document contains a title, uid, and some text content. \
Use this context to answer the question. \
If the answer is not in the context or history, say you don't know. \
Keep answers concise and to the point — ideally within three sentences.

    {context}"""
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    return qa_prompt



