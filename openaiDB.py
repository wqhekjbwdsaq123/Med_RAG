import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import CSVLoader
from langchain_text_splitters import CharacterTextSplitter

# 1. API KEY 설정 (OpenAI)
load_dotenv()
# os.environ["OPENAI_API_KEY"] = "sk-..." # API 키는 .env 파일에 저장하세요.

# 2. CSV 파일 로드
# CSV 파일 경로를 지정하세요. 
# 만약 파일에 특정 인코딩이 필요하면 encoding='utf-8' 등을 추가하세요.
print("CSV 파일 로드 중...")
loader = CSVLoader(
    file_path=r"C:\Users\asia\Downloads\archive\mtsamples.csv",
    encoding='utf-8' # 이 부분을 추가해야 합니다.
)
documents = loader.load()

# 3. 텍스트 분할 (Splitter)
print("텍스트 분할 중...")
# CSV의 경우 구조에 따라 문장이 너무 길 수 있으니 조정 필요
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(documents)

# 4. OpenAI 임베딩 설정
print("OpenAI 임베딩 생성 중... (비용 발생)")
embeddings = OpenAIEmbeddings() 

# 5. FAISS 벡터 데이터베이스 생성 및 저장
print("FAISS 데이터베이스 생성 중...")
vector_store = FAISS.from_documents(docs, embeddings)
vector_store.save_local("vectorstore") # 기존 폴더 덮어쓰기
print("FAISS 데이터베이스 생성 완료: 'vectorstore' 폴더")