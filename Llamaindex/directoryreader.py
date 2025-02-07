from llama_index.readers import SimpleDirectoryReader
from llama_index import VectorStoreIndex
import os
from dotenv import load_dotenv
import logging
import sys

load_dotenv()

def main(url:str) -> None:
    documents = SimpleDirectoryReader(url).load_data()
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine()
    response = query_engine.query("What did the artical saying please summarize it?")
    print(response)
    
    
if __name__ == "__main__":
    main(url=r"C:\Users\vikas\OneDrive\Desktop\Generative-AI-Sunny-Sir\Llamaindex\data")    
