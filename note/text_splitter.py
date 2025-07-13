# chatbot/langchain_pipeline/text_splitter.py
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document

import re
from langchain.text_splitter import RecursiveCharacterTextSplitter

def group_caption_blocks(content: str) -> list[str]:
    """
    Groups lines like:
    [Image 1 Caption]: ...
    [Image Path]: ...
    into single blocks, so they're not split during chunking.
    """

    blocks = []
    current_block = []
    lines = content.splitlines()

    caption_pattern = re.compile(r"^\[Image\s+\d+\s+Caption\]:", re.IGNORECASE)

    for i, line in enumerate(lines):
        if caption_pattern.match(line.strip()):
            # Flush previous block if any
            if current_block:
                blocks.append("\n".join(current_block).strip())
                current_block = []
        
        current_block.append(line)

        # Also include [Image Path]: line right after caption
        if i + 1 < len(lines) and lines[i + 1].strip().startswith("[Image Path]:"):
            current_block.append(lines[i + 1])
            # Skip the next line from loop
            lines[i + 1] = ""  # Avoid duplication

    if current_block:
        blocks.append("\n".join(current_block).strip())

    # Any leftover lines not in a block
    leftovers = [line.strip() for line in lines if line.strip() and not any(line.strip() in b for b in blocks)]
    if leftovers:
        blocks.extend(leftovers)

    return blocks


def split_text_preserving_blocks(content: str, chunk_size=800, chunk_overlap=100) -> list[str]:
    """
    Use LangChain splitter, but protect caption blocks from being split.
    """
    paragraphs = group_caption_blocks(content)
    text = "\n\n".join(paragraphs)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        #separators=["\n\n", "\n", ".", " "],
        is_separator_regex=False
    )
    return splitter.split_text(text)

def manual_chunk_paragraphs(blocks: list[str], chunk_size: int = 800) -> list[str]:
    chunks = []
    current = ""
    for block in blocks:
        if len(current) + len(block) < chunk_size:
            current += "\n\n" + block
        else:
            chunks.append(current.strip())
            current = block
    if current:
        chunks.append(current.strip())
    return chunks


def split_text(text: str, chunk_size=800, chunk_overlap=100) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)

def split_with_overlap_preserving_blocks(
    content: str,
    chunk_size: int = 800,
    chunk_overlap: int = 100
) -> list[str]:
    blocks = group_caption_blocks(content)

    chunks = []
    current_chunk = []
    current_len = 0

    for block in blocks:
        block_len = len(block)

        if current_len + block_len > chunk_size and current_chunk:
            # Save current chunk
            chunks.append("\n\n".join(current_chunk).strip())

            # Start overlap chunk (last blocks until overlap size)
            overlap_text = "\n\n".join(current_chunk)[-chunk_overlap:]
            current_chunk = [overlap_text]
            current_len = len(overlap_text)

        current_chunk.append(block)
        current_len += block_len

    if current_chunk:
        chunks.append("\n\n".join(current_chunk).strip())

    return chunks



def chunk_note(note_uid: str, title: str, content: str) -> tuple[list[Document], list[str]]:
    chunks = split_text(content)
    documents = []
    ids = []

    for i, chunk in enumerate(chunks):
        doc = Document(
            page_content = f"[Title: {title}] [UID: {note_uid}]\n{chunk}",
            metadata={
                "uid": note_uid,
                "title": title,
                "chunk_id": i,
                "contains_image": "[Image Path]:" in chunk or "[Image" in chunk
            }
        )
        documents.append(doc)
        ids.append(f"{note_uid}_chunk_{i}")
    
    return documents, ids

