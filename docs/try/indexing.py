# langchain을 사용하여 디렉토리에서 문서를 로드합니다.
## DirectoryLoader 는 디렉토리를 입력으로 받고 해당 디렉토리에 있는 모든 문서를 로드
from langchain.document_loaders import DirectoryLoader
 
directory = '/content/data'
 
def load_docs(directory):
 loader = DirectoryLoader(directory)
 documents = loader.load()
 return documents
 
documents = load_docs(directory)
len(documents)

# 문서 분리
# 문서를 로드한 후 스크립트는 이러한 문서를 더 작은 청크로 분리함 -> RecursiveCharacterTextSplitter 사용
from langchain.text_splitter import RecursiveCharacterTextSplitter
 
def split_docs(documents,chunk_size=500,chunk_overlap=20):
 text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
 docs = text_splitter.split_documents(documents)
 return docs
 
docs = split_docs(documents)
print(len(docs))``

# 임베딩 생성
## 문서 분리 후 AI모델이 이해할 수 있는 형식으로 데이터 청크를 변환함 ->  SentenceTransformerEmbeddings 사용
from langchain.embeddings import SentenceTransformerEmbeddings
 
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Pinecone에 임베딩 저장
## 임베딩이 생성된 후, 쉽게 엑세스하고 검색할 수 있는 곳에 저장하기.

from langchain.pinecone import PineconeIndexer
 
def index_embeddings(embeddings, docs):
    indexer = PineconeIndexer(api_key='your-pinecone-api-key', index_name='your-index-name')
    indexer.index(embeddings, docs)
 
index_embeddings(embeddings, docs)

## 이 스크립트는 Pinecone에서 인덱스를 생성하고 임베딩을 해당 텍스트와 함께 저장사용자가 질문을 하면 chatbot은 이 인덱스에서 가장 유사한 텍스트를 검색하고 해당하는 답변을 반환할 수 있습니다.