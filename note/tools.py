from langchain_core.tools import tool
from .llm_instance import llm
from .vectorestore import get_vectorstore

@tool
def create_note_from_prompt(prompt: str) -> dict:
    """
    Creates a note from a natural language prompt.
    Returns a dict with keys: title, content, or an error message.
    """

    system_instruction = """
You are an intelligent note-creating agent.

Your task is to generate a concise and useful note based on the user instruction below.

You may:
- Use previous assistant messages if helpful.
- Search the knowledge base (retrieved note chunks).
- Summarize one or more existing notes.
- Synthesize insights from related topics.

You decide how to best serve the request.

Respond only in the format:
title: <title of the note>
content: <note content>
"""

    full_prompt = f"{system_instruction}\n\nUser prompt:\n{prompt}"
    response = llm.invoke(full_prompt)

    content = getattr(response, "content", str(response))
    lines = content.strip().splitlines()

    title = ""
    note_content = ""

    for line in lines:
        if line.lower().startswith("title:"):
            title = line.split(":", 1)[1].strip()
        elif line.lower().startswith("content:"):
            note_content = line.split(":", 1)[1].strip()

    if not title or not note_content:
        return {"error": "Failed to generate a note with title and content."}

    return {"title": title, "content": note_content}


@tool
def open_note_tool(query: str) -> dict:
    """
    Use vectorstore similarity search to find the most relevant note chunk(s) for the query.
    Returns a dictionary with the best matching note info.
    """

    try:
        vectorstore = get_vectorstore()
        results = vectorstore.similarity_search(query, k=3)

        if not results:
            return {"note_found": False, "error": "No relevant notes found."}

        top_doc = results[0]

        title = top_doc.metadata.get("title", "Untitled")
        content = top_doc.page_content

        return {
            "note_found": True,
            "note_title": title,
            "note_content": content,
            "note_uid": top_doc.metadata.get("uid", "")
        }

    except Exception as e:
        return {"note_found": False, "error": f"Error retrieving notes: {str(e)}"}
