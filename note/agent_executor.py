from langchain.agents import AgentExecutor, ZeroShotAgent
from langchain.agents import load_tools
from .tools import create_note_from_prompt, open_note_tool, search_note_tool
from .llm_instance import llm
from langchain.chains import LLMChain

def get_agent_executor():
    tools = [create_note_from_prompt, open_note_tool, search_note_tool]

    prefix = """
You are a helpful assistant that can perform specific note-related actions.

You have access to these tools:

1. create_note_from_prompt:
   - Use this when the user wants to create a new note based on input, a summary, or the current conversation.

2. open_note_tool:
   - Use this when the user wants to open or retrieve a specific note by its title or based on a topic.

3. search_note_tool:
   - Use this when the user wants to search or explore a list of notes related to a topic, tag, or keyword.

Only use a tool if the user explicitly requests one of these actions.
Otherwise, the assistant (chatbot) will answer normally without calling any tools.
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
