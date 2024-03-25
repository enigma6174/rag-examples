import os

from pymongo import MongoClient
from dotenv import load_dotenv

from langchain_mongodb.vectorstores import MongoDBAtlasVectorSearch

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.chat_models.ollama import ChatOllama
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.azure_cosmos_db import (
    AzureCosmosDBVectorSearch,
    CosmosDBSimilarityType,
    CosmosDBVectorSearchType
)

from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# load the environment variable CONNECTION_STRING to connect to MongoDB cluster
CONNECTION_STRING = os.environ.get("COSMOS_MONGODB_STRING")
print(f"CONNECTION_STRING: {CONNECTION_STRING}\n\n")

# define the parameters - database name, collection name, search index key
INDEX_NAME = "sample-index"
NAMESPACE = "sample-db.sample-collection"
DB_NAME, COLLECTION_NAME = NAMESPACE.split(".")


# load the PDF document
loader = PyPDFLoader("https://arxiv.org/pdf/2303.08774.pdf")
data = loader.load()

# split the document into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
docs = text_splitter.split_documents(data)

print(docs[0])
print("\n\n")

# local embedding model: all-mpnet-base-v2 (HuggingFace)
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# local chat model: llama2-chat-13b (Ollama)
llm = ChatOllama(model="llama2:13b-chat")

# create the MongoDB client
client = MongoClient(CONNECTION_STRING)

# define the collection name
DB_COLLECTION = client[DB_NAME][COLLECTION_NAME]

print("=> preparing the collection...")

# insert the documents into the collection with their embeddings
try:
    vector_search = MongoDBAtlasVectorSearch.from_documents(
        documents=docs,
        embedding=embedding,
        collection=DB_COLLECTION,
        index_name=INDEX_NAME
    )
    print("=> collection prepared successfully...")
except Exception as e:
    print(f"[ERR]\n{e}")

# get the vector store
try:
    vectorstore = MongoDBAtlasVectorSearch.from_connection_string(
        connection_string=CONNECTION_STRING,
        namespace=NAMESPACE,
        embedding=embedding,
    )
    print("=> connected to vectorstore")
except Exception as e:
    print(f"[ERR]\n{e}")
