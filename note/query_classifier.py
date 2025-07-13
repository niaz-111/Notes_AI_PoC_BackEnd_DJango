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


def is_agent_task_precise(text: str) -> bool:
    """
    Use LLM to classify whether a user input is a task that requires agent execution:
    - Open a note
    - Create a note with content
    - Search notes by topic/title/content

    If none of these, return NO (chat task).
    """

    system_prompt = """
You are a task classifier for a notes assistant. Your job is to decide whether a user's input requires agent execution or is just a normal chat question.

Only classify the input as an agent task (respond "YES") if the user clearly intends to do **one** of the following:

1. 🟩 **Open a note**
   - e.g., "Open the note titled 'Meeting Plan'"
   - e.g., "Show me the note about Marketing"

2. 🟩 **Create a note with content**
   - e.g., "Create a note from this summary"
   - e.g., "Save this as a note"

3. 🟩 **Search notes**
   - e.g., "Find notes about travel"
   - e.g., "Search my notes for climate change"

If the input does **not clearly match** one of those 3 types (even if it's related to notes), classify it as a **chat task (respond "NO")**.

🟥 Do **NOT** classify the following as agent tasks:
- Asking about the content of a note (e.g., "What does the marketing note say?")
- Asking if you know about a note (e.g., "Do you know about the note titled 'Budget'?")
- Asking for summaries or explanations (e.g., "Summarize the productivity note")

Answer only with:
- "YES" → if it's an agent task (open/create/search note)
- "NO" → for all other chat or informational queries

User input:
{text}

Answer:
"""
    prompt = f"{system_prompt}\nUser input:\n{text}\nAnswer:"
    response = llm.invoke(prompt)
    answer = getattr(response, "content", str(response)).strip().upper()

    return answer == "YES"





