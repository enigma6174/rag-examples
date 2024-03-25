import os

from dotenv import load_dotenv

from langchain_community.chat_models.ollama import ChatOllama
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.azure_cosmos_db import AzureCosmosDBVectorSearch

load_dotenv()

# load the environment variable CONNECTION_STRING to connect to MongoDB cluster
CONNECTION_STRING = os.environ.get("COSMOS_VCORE_STRING")
print(CONNECTION_STRING)

# define the parameters - database name, collection name, search index key
INDEX_NAME = "sample-index"
NAMESPACE = "sample-db.sample-collection"
DB_NAME, COLLECTION_NAME = NAMESPACE.split(".")

# local embedding model: all-mpnet-base-v2 (HuggingFace)
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# local chat model: llama2-chat-13b (Ollama)
llm = ChatOllama(model="llama2:13b-chat")

# get the vector store
vector_store = AzureCosmosDBVectorSearch.from_connection_string(
    connection_string=CONNECTION_STRING,
    namespace=NAMESPACE,
    embedding=embedding,
    index_name="vsearch_index"
)
print("=> connected to vectorstore\n")

query = "What did the president say about Afghanistan?"
results = vector_store.similarity_search(query)

print(results[0].page_content)

