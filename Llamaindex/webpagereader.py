#from llama_index.llms import OpenAI
#from llama_index.readers import SimpleWebPageReader
from llama_index.core import SimpleDirectoryReader
#from llama_index import VectorStoreIndex
from llama_index.core.vector_stores import ChromaVectorStore
import chromadb
import llama_index
import os
from dotenv import load_dotenv

load_dotenv()

def main(url:str) -> None:
    document = SimpleDirectoryReader(html_to_text = True).load_data(url = [url])
    index = ChromaVectorStore.from_documents(documents = document)
    query_engine = index.as_query_engine()
    response = query_engine.query("What is Genrative AI?")
    print(response)
    
if __name__ == "__main__":
    main(url="https://medium.com/@social_65128/the-comprehensive-guide-to-understanding-generative-ai-c06bbf259786")    

