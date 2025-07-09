from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from .qa_chain import get_conversational_rag_chain
from .llm_instance import llm

from tools import (
    create_note_from_input,
    open_note_from_query,
    get_last_llm_result
)
  


router_prompt = PromptTemplate.from_template("""
You are an intelligent router. Classify the user's intent from their input.

If the user wants to:
- Create a new note, return: create_note
- Open a note, return: open_note
- Otherwise, return: chat

Input: {input}
""")

classifier_chain = router_prompt | llm | StrOutputParser()


def handle_chat_request(user_input: str, session_id: str) -> str:
    def route_action(classification: str):
        if "create_note" in classification:
            return RunnableLambda(lambda x: create_note_from_input(x["input"], session_id))
        elif "open_note" in classification:
            return RunnableLambda(lambda x: open_note_from_query(x["input"]))
        else:
            rag_chain = get_conversational_rag_chain()
            return RunnableLambda(lambda x: rag_chain.invoke({
                "input": x["input"],
                "chat_history": [],
            }, config={"configurable": {"session_id": x["session_id"]}}))

    chain = (
        {"input": lambda x: x["input"]}
        | classifier_chain
        | RunnableBranch(
            (lambda cls: "create_note" in cls, route_action("create_note")),
            (lambda cls: "open_note" in cls, route_action("open_note")),
            route_action("chat")
        )
    )

    return chain.invoke({"input": user_input, "session_id": session_id})
