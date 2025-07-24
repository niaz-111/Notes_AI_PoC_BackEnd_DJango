from langchain_ollama.llms import OllamaLLM
from langchain_ollama import OllamaEmbeddings
import os
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI

#OLLAMA_URL = "http://107.109.214.240:11434/api/generate"
OLLAMA_URL = "http://127.0.0.1:11434"

#deepseek v2 local
#llm = OllamaLLM(model="deepseek-v2", temperature=0, base_url=OLLAMA_URL)

#gemini flash 1.5 API key
# llm = ChatGoogleGenerativeAI(
#     model="models/gemini-1.5-flash-latest",  
#     google_api_key="AIzaSyDRFIqQrekDkbULaXo8TbIrdZSz7P7TPMk",     
#     temperature=0.3,
#     convert_system_message_to_human=True
# )

# AIzaSyDRFIqQrekDkbULaXo8TbIrdZSz7P7TPMk
# AIzaSyD2QD_eASRj1mdZKliFXIfgPKh253KTmBE # expired
# AIzaSyDHDWYS1aRkHx9gAoGYi0Ya1M5wAh9dlMM
# AIzaSyCktwsTXH2pX8mBYuIp_ork0uaWy3WWWIM - azwad
# AIzaSyCCWL591ROexX76B8tqCk-Vr9EVOtAkEHo - hasan
# AIzaSyD0ruej5ud-MvX6vlItbrwT-hNKHOXt8XY - asif

#gemini flash 2.5 API key
llm = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash",  
    google_api_key="AIzaSyCCWL591ROexX76B8tqCk-Vr9EVOtAkEHo",     
    temperature=0,
    convert_system_message_to_human=True
)

embedding_model = OllamaEmbeddings(model="nomic-embed-text", base_url=OLLAMA_URL)