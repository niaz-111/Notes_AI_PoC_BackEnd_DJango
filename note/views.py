from rest_framework.decorators import api_view, parser_classes
from rest_framework.response import Response
from rest_framework import status
from rest_framework.parsers import MultiPartParser
from .models import Note
from .serializers import NoteSerializer
from .pipeline_runner import run_ingestion_pipeline
from .vectorestore import delete_note_by_uid
from .qa_chain import get_conversational_rag_chain
from .vectorestore import get_vectorstore
from .qa_chain import get_retrieval_qa_chain
from .chat_session_manager import clear_all_sessions
from .agent_executor import get_agent_executor, get_agent_executor_with_history
from .query_classifier import is_agent_task,is_agent_task_precise
from .image_description_generator import generate_image_caption, generate_gemini_caption
from langchain_core.exceptions import OutputParserException
import json

@api_view(['POST'])
@parser_classes([MultiPartParser])
def upload_note_with_file(request):
    uid = request.POST.get('uid')
    title = request.POST.get('title')

    if not uid or not title:
        return Response({"error": "uid, title, and text_file are required."}, status=status.HTTP_400_BAD_REQUEST)

    text_file = request.FILES.get('text_file')
    text_content = request.POST.get('text_content')

    full_content = ""

    try:
        if text_file:
            if not text_file.name.endswith('.txt'):
                return Response({"error": "Only .txt files are accepted."}, status=status.HTTP_400_BAD_REQUEST)
            text = text_file.read().decode('utf-8')
        elif text_content:
            full_content = text_content
        else:
            note_parts = []
            i = 0
            image_index = 0
            while True:
                if f"part_{i}_type" not in request.POST:
                    break

                part_type = request.POST.get(f"part_{i}_type")
                if part_type == "text":
                    text = request.POST.get(f"part_{i}_text", "")
                    note_parts.append(f"[Text]: {text.strip()}")
                elif part_type == "image":
                    path = request.POST.get(f"part_{i}_image_path", "")
                    #caption = generate_image_caption(path)  # Local captioning
                    caption = generate_gemini_caption(path)  # Gemini captioning
                    note_parts.append(f"[Image {image_index+1} Caption]: {caption}\n[Image Path]: {path}")
                    image_index += 1
                i += 1

            full_content = "\n\n".join(note_parts)

        
        delete_note_by_uid(uid)
        run_ingestion_pipeline([{
            "uid": uid,
            "title": title,
            "content": full_content
        }])

        return Response("ok", status=201)

    except Exception as e:
        return Response({"error": str(e)}, status=500)


# Placeholder for actual logic
# from .chat_engine import get_answer_from_langchain


@api_view(['POST'])
def ask_question_agent(request):
    user_input = request.data.get('question')
    if not user_input:
        return Response({"error": "Question is required."}, status=status.HTTP_400_BAD_REQUEST)

    try:
        if is_agent_task_precise(user_input):
            agent_executor = get_agent_executor_with_history()

            try:
                result = agent_executor.invoke(
                    {"input": user_input},
                    config={"configurable": {"session_id": "abc123"}}
                )
            except OutputParserException as e:
                # LLM didn't call a tool or produced a natural reply
                return Response({
                    "type": "agent",
                    "answer": str(e)  # fallback as plain message
                }, status=status.HTTP_200_OK)

            # Extract agent's output (Final Answer)
            output = result["output"] if isinstance(result, dict) else str(result)

            # Parse the expected JSON tool output
            try:
                parsed = json.loads(output)

                tool_name = parsed.get("tool")
                payload = parsed.get("payload")

                if not tool_name or payload is None:
                    # Malformed result, fallback to showing raw
                    return Response({
                        "type": "agent",
                        "answer": output
                    }, status=status.HTTP_200_OK)

                return Response({
                    "type": "agent",
                    "agent_data": {
                        "tool": tool_name,
                        "payload": payload
                    }
                }, status=status.HTTP_200_OK)

            except json.JSONDecodeError:
                # Tool output was not valid JSON
                return Response({
                    "type": "agent",
                    "answer": output
                }, status=status.HTTP_200_OK)

        else:
            conversational_rag_chain = get_conversational_rag_chain()
            response = conversational_rag_chain.invoke({"input": user_input}, config={"configurable": {"session_id": "abc123"}})
            answer = response.get("answer", "Sorry, I don't know the answer.")
            return Response({"type": "chat", "answer": answer}, status=status.HTTP_200_OK)

    except Exception as e:
        return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


def infer_tool_from_output(parsed: dict) -> str:
    if not isinstance(parsed, dict):
        return "unknown_tool"

    if "note_uid" in parsed and parsed.get("note_found") is not False:
        return "open_note_tool"
    elif "notes" in parsed:
        return "search_note_tool"
    elif "title" in parsed and "content" in parsed:
        return "create_note_from_prompt"
    return "unknown_tool"


@api_view(['POST'])
def ask_question(request):
    question = request.data.get('question')

    if not question:
        return Response({"error": "Question is required."}, status=status.HTTP_400_BAD_REQUEST)

    try:
        conversational_rag_chain = get_conversational_rag_chain()

        answer = conversational_rag_chain.invoke(
            {"input": question},
            config={
            "configurable": {"session_id": "abc123"}
            },
            )["answer"]

        answer = f"{answer}"

        return Response({"answer": answer}, status=status.HTTP_200_OK)

    except Exception as e:
        return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    

@api_view(['GET'])
def clear_notes(request):
    try:
        vectorstore = get_vectorstore()
        count = vectorstore._collection.count()

        vectorstore.delete_collection()

        return Response({"message": f"Deleted {count} chunks."}, status=status.HTTP_200_OK)
    except Exception as e:
        return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['POST'])
def retrieval_chain_qa(request):
    question = request.data.get('question')

    if not question:
        return Response({"error": "Question is required."}, status=status.HTTP_400_BAD_REQUEST)

    try:
        qa_chain = get_retrieval_qa_chain()
        result = qa_chain({"query": question})

        return Response({"answer": result["result"]}, status=status.HTTP_200_OK)
    except Exception as e:
        return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    

@api_view(['GET'])
def clear_all_hat_sessions_api(request):
    clear_all_sessions()
    return Response({"message": "All chat sessions cleared."}, status=status.HTTP_200_OK)