from langchain_ollama.llms import OllamaLLM
from langchain_ollama import OllamaEmbeddings

#OLLAMA_URL = "http://107.109.214.240:11434/api/generate"
OLLAMA_URL = "http://127.0.0.1:11434"

print("LLM model is initializing")
llm = OllamaLLM(model="deepseek-v2", temperature=0, base_url=OLLAMA_URL)
embedding_model = OllamaEmbeddings(model="nomic-embed-text", base_url=OLLAMA_URL)