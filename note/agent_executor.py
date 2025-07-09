from langchain.agents import AgentExecutor, ZeroShotAgent
from langchain.agents import load_tools
from .tools import create_note_from_prompt, open_note_tool
from .llm_instance import llm
from langchain.chains import LLMChain

def get_agent_executor():
    tools = [create_note_from_prompt, open_note_tool]

    # Instructions to guide the agent on when/how to use tools
    prefix = """
You are a helpful assistant that can perform note-related tasks.

You can:
- create_note_from_prompt: to create a new note based on a prompt.
- open_note_tool: to open the most relevant note for a query.

When user input suggests these actions, use the appropriate tool.

Respond with tool calls ONLY when appropriate.
"""

    suffix = """
Begin!

User input: {input}
{agent_scratchpad}
"""

    # Create the prompt for the agent
    prompt = ZeroShotAgent.create_prompt(
        tools=tools,
        prefix=prefix,
        suffix=suffix,
        input_variables=["input", "agent_scratchpad"],
    )

    # Wrap the ChatGoogleGenerativeAI instance in an LLMChain
    llm_chain = LLMChain(
        llm=llm,  # Assuming `llm` is defined globally or passed in
        prompt=prompt,
    )

    # Now create the agent using the proper chain
    agent = ZeroShotAgent(
        llm_chain=llm_chain,
        allowed_tools=[tool.name for tool in tools],
    )

    # Create the executor with tools and the agent
    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors = True
    )

    return agent_executor
