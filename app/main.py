import os

from dotenv import load_dotenv
from pymongo import MongoClient

from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.chat_models.ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.azure_cosmos_db import AzureCosmosDBVectorSearch

load_dotenv()

CONNECTION_STRING = os.environ.get("COSMOS_VCORE_STRING")
print(CONNECTION_STRING)
print()

INDEX_NAME = "facts-test-index"
NAMESPACE = "facts_test_db.test_collection"
DB_NAME, COLLECTION_NAME = NAMESPACE.split(".")

# local embedding model: all-mpnet-base-v2 (HuggingFace)
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# local chat model: llama2-chat-13b (Ollama)
llm = ChatOllama(model="llama2:13b-chat")

# define the client for CosmosDb
client: MongoClient = MongoClient(CONNECTION_STRING)

# define the collection
collection = client[DB_NAME][COLLECTION_NAME]
print(collection)

# create a retriever using the vector store
vectorstore = AzureCosmosDBVectorSearch.from_connection_string(
    connection_string=CONNECTION_STRING,
    namespace=NAMESPACE,
    embedding=embedding,
)
retriever = vectorstore.as_retriever()

try:
    query = "Does ostrich have a small brain?"
    docs = vectorstore.similarity_search(query, k=5)
except Exception as e:
    print(f"[ERR] {e}")
else:
    for doc in docs:
        print(doc)
        print(doc.metadata)


# design the chat prompt
prompt = ChatPromptTemplate.from_template("{question}")

# create conversation chain
chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")

# test the system
while True:
    question = input(">> ")
    result = chain.invoke({'query': question})
    print(result)
