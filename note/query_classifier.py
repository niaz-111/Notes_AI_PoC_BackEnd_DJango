from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from .llm_instance import llm

router_prompt = PromptTemplate.from_template("""
Classify the user input into one of the following categories:
- agent: if the user is trying to create, open, or manage notes
- rag: if the user is asking a question that should be answered from existing notes

Input: {input}
Answer (only write 'agent' or 'rag'):
""")

classifier_chain = router_prompt | llm | StrOutputParser()

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableMap
from langchain_core.output_parsers import StrOutputParser
from .llm_instance import llm 

intent_prompt = ChatPromptTemplate.from_messages([
    ("system", 
     """You are an intent classifier. 
     You will receive a user's input and must classify it as one of the following intents:
     1. general_qa
     2. create_note
     3. open_note

     Return ONLY the intent name, nothing else.
     """),
    ("human", "{input}")
])

intent_classifier_chain = intent_prompt | llm | StrOutputParser()

def classify_intent(user_input: str) -> str:
    return intent_classifier_chain.invoke({"input": user_input}).strip().lower()


def is_agent_task(text: str) -> bool:
    """
    Use LLM to classify if user input requires agent (tool) execution or is a normal chat message.

    Returns True if agent task, False otherwise.
    """

    system_prompt = """
You are a classifier. Determine if the following user input requires executing special tools (like creating or opening notes)
or if it is a normal chat question.

Return only "YES" if it's an agent task, else "NO".
"""

    prompt = f"{system_prompt}\nUser input:\n{text}\nAnswer:"

    response = llm.invoke(prompt)
    answer = getattr(response, "content", str(response)).strip().upper()

    return answer == "YES"




