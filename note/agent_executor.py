from langchain.agents import AgentExecutor, ZeroShotAgent
from langchain.agents import load_tools
from .tools import create_note_from_prompt, open_note_tool, search_note_tool
from .llm_instance import llm
from langchain.chains import LLMChain

def get_agent_executor():
    tools = [create_note_from_prompt, open_note_tool, search_note_tool]

    prefix = """
You are a helpful assistant that can perform note-related actions using tools.

You have access to these tools:

1. create_note_from_prompt:
   - Use this when the user wants to create a new note from a summary or message.

2. open_note_tool:
   - Use this to open and retrieve a specific note using a title or topic.

3. search_note_tool:
   - Use this to search for a list of notes related to a keyword or topic.

Important Rules:
- When you decide to use a tool, only call the tool and return its **exact output as-is**.
- The tools return JSON-formatted strings. Do **not** change, summarize, or explain them.
- Do **not** add any comments like “Here is the note…” or “Okay, I found it…” — just return the raw output of the tool.
- If none of the tools match the intent, do nothing and let the chatbot handle the input instead.
"""

    suffix = """
Begin!

User input: {input}
{agent_scratchpad}
"""

    prompt = ZeroShotAgent.create_prompt(
        tools=tools,
        prefix=prefix,
        suffix=suffix,
        input_variables=["input", "agent_scratchpad"],
    )

    llm_chain = LLMChain(
        llm=llm,  # Your global or injected LLM (e.g., Gemini, GPT-4, etc.)
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
        handle_parsing_errors=True
    )

    return agent_executor
