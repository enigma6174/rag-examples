import os

from dotenv import load_dotenv
from pymongo import MongoClient

from langchain_community.chat_models.ollama import ChatOllama

from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.azure_cosmos_db import (
    AzureCosmosDBVectorSearch,
    CosmosDBSimilarityType,
    CosmosDBVectorSearchType
)

from langchain_text_splitters import CharacterTextSplitter

load_dotenv()

SOURCE_FILE_NAME = "data/state_of_union.txt"

CONNECTION_STRING = os.environ.get("COSMOS_VCORE_STRING")
print(CONNECTION_STRING)

INDEX_NAME = "sample-index"
NAMESPACE = "sample-db.sample-collection"
DB_NAME, COLLECTION_NAME = NAMESPACE.split(".")

# load the documents into memory
loader = TextLoader(SOURCE_FILE_NAME)
documents = loader.load()

# split the documents into chunks
text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

# local embedding model: all-mpnet-base-v2 (HuggingFace)
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# local chat model: llama2-chat-13b (Ollama)
llm = ChatOllama(model="llama2:13b-chat")

# define the client for CosmosDb
client: MongoClient = MongoClient(CONNECTION_STRING)

# define the collection
collection = client[DB_NAME][COLLECTION_NAME]

# define the vector store
vectorstore = AzureCosmosDBVectorSearch.from_documents(
    docs,
    embedding,
    collection=collection,
    index_name=INDEX_NAME
)

# parameters for Azure CosmosDb (mongo)
num_lists = 100
dimensions = 768
similarity_algorithm = CosmosDBSimilarityType.COS
kind = CosmosDBVectorSearchType.VECTOR_IVF
m = 16
ef_construction = 64
ef_search = 40
score_threshold = 0.1

# create the vector store
try:
    vectorstore.create_index(
        num_lists, dimensions, similarity_algorithm, kind, m, ef_construction
    )
except Exception as e:
    print(f"[ERR] {e}")
else:
    print("Index Created Successfully!")
