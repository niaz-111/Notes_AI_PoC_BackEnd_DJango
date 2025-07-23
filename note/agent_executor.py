from langchain.agents import AgentExecutor, ZeroShotAgent
from langchain.agents import load_tools
from .tools import create_note_from_prompt, open_note_tool, search_note_tool
from .llm_instance import llm
from .chat_session_manager import get_session_history
from langchain.chains import LLMChain
from langchain_core.runnables.history import RunnableWithMessageHistory

def get_agent_executor():
    tools = [create_note_from_prompt, open_note_tool, search_note_tool]

    for tool in tools:
        print(tool.name)


    prefix = """
You are a helpful assistant that can perform note-related actions using tools.

You have access to these tools:

1. create_note_from_prompt:
   - Use this when the user wants to create a new note from a summary or message  or prior assistant response.
   - If the user says “using this/it/that/last response”, etc., assume they are referring to the last assistant message in the chat history. Include that message in the tool input to give full context.

2. open_note_tool:
   - Use this to open and retrieve a specific note using a title or topic.

3. search_note_tool:
   - Use this to search for a list of notes related to a keyword or topic.

Important Rules:
- Use chat history to resolve references like “this note”, “that one”, or “the note we discussed earlier”.
- Only take action if the request is clearly about creating, opening, or searching notes.
- When you decide to use a tool, only call the tool and return its **exact output as-is**.
- The tools return JSON-formatted strings. Do **not** change, summarize, or explain them.
- Do **not** add any comments like “Here is the note…” or “Okay, I found it…” — just return the raw output of the tool.
- If none of the tools match the intent, do nothing and let the chatbot handle the input instead.
- If the user's request and chat history matches a tool but lacks enough detail to call it (e.g., missing note title, topic etc), respond in natural language asking for clarification.

AFTER receiving a tool result (Observation), you must follow this format:

Thought: [reflect on the tool result]
Final Answer: [return the exact tool result or answer clearly]

Never return raw JSON after Thought: — it must always be under Final Answer.
Never skip Final Answer if you're done.
"""

    suffix = """
Chat history:
{chat_history}

Begin!

User input: {input}
{agent_scratchpad}
"""

    prompt = ZeroShotAgent.create_prompt(
        tools=tools,
        prefix=prefix,
        suffix=suffix,
        input_variables=["input", "agent_scratchpad", "chat_history"],
    )

    llm_chain = LLMChain(
        llm=llm,
        prompt=prompt,
    )

    agent = ZeroShotAgent(
        llm_chain=llm_chain,
        allowed_tools=[tool.name for tool in tools],
    )

    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=3,  # Prevent infinite loops
        early_stopping_method="generate"  # Graceful fallback
    )

    return agent_executor


def get_agent_executor_with_history():
    agent_executor = get_agent_executor()

    return RunnableWithMessageHistory(
        agent_executor,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )
