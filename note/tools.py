from langchain_core.tools import tool
from .llm_instance import llm
from .vectorestore import get_vectorstore
import json

# === Tool 1: Create Note from Prompt ===
@tool
def create_note_from_prompt(prompt: str) -> str:
    """
    Creates a note from a natural language prompt.
    Returns a JSON string with keys: title, content.
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
        return json.dumps({"error": "Failed to generate a note with title and content."})

    return json.dumps({"title": title, "content": note_content})


# === Tool 2: Open Note by Query ===
@tool
def open_note_tool(query: str) -> str:
    """
    Use vectorstore similarity search to find the most relevant note.
    Input: query string (title or topic)
    Output: JSON with note_found, title, content, uid or error.
    """
    try:
        vectorstore = get_vectorstore()
        results = vectorstore.similarity_search(query, k=3)

        if not results:
            return json.dumps({"note_found": False, "error": "No relevant notes found."})

        top_doc = results[0]

        return json.dumps({
            "note_found": True,
            "note_title": top_doc.metadata.get("title", "Untitled"),            
            "note_uid": top_doc.metadata.get("uid", "")
        })

    except Exception as e:
        return json.dumps({"note_found": False, "error": f"Error retrieving notes: {str(e)}"})


# === Tool 3: Search Notes ===
@tool
def search_note_tool(query: str) -> str:
    """
    Searches for notes by topic, title, or content keywords.
    Input: query string
    Output: JSON with list of matching note previews.
    """
    try:
        vectorstore = get_vectorstore()
        results = vectorstore.similarity_search(query, k=10)

        if not results:
            return json.dumps({
                "matches_found": False,
                "notes": [],
                "error": "No relevant notes found."
            })

        notes = []
        for doc in results:
            notes.append({
                "note_title": doc.metadata.get("title", "Untitled"),
                "note_uid": doc.metadata.get("uid", ""),                
            })

        return json.dumps({
            "matches_found": True,
            "query": query,
            "notes": notes
        })

    except Exception as e:
        return json.dumps({
            "matches_found": False,
            "notes": [],
            "error": f"Error searching notes: {str(e)}"
        })

