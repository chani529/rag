import os
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.prompts import load_prompt
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.documents import Document
import logging
import traceback
import pandas as pd
import uuid

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 환경 변수 설정
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "ggml-model-Q5_K_M")
CACHE_DIR = ".cache/embeddings"
FAISS_INDEX_DIR = ".cache/files"

app = FastAPI(title="100% 오픈모델 RAG API")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    question: str
    session_id: str

class ChatResponse(BaseModel):
    answer: str

class UploadResponse(BaseModel):
    session_id: str
    message: str

# 세션별 체인 저장
chains: Dict[str, object] = {}

def create_chain(vectorstore):
    """벡터스토어를 사용하여 체인 생성"""
    try:
        # 문서 검색기 설정
        retriever = vectorstore.as_retriever()

        # 프롬프트 로드
        prompt = load_prompt("prompts/rag-exaone.yaml", encoding="utf-8")

        # Ollama 모델 지정
        llm = ChatOllama(
            model=f"{EMBEDDING_MODEL}:latest",
            temperature=0,
        )

        # 체인 생성
        chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        logger.info("Chain created successfully")
        return chain
    except Exception as e:
        logger.error(f"Error creating chain: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def process_csv_file(file_path: str):
    """CSV 파일 처리 및 벡터스토어 생성"""
    try:
        # CSV 파일 읽기
        df = pd.read_csv(file_path)
        
        # 문서 생성
        docs = []
        for _, row in df.iterrows():
            content = f"category: {row['category']}\ntype: {row['type']}\ntitle: {row['title']}\ncontent: {row['content']}"
            doc = Document(
                page_content=content,
                metadata={
                    "source": file_path,
                    "category": row['category'],
                    "type": row['type'],
                    "title": row['title']
                }
            )
            docs.append(doc)

        # Splitter 설정
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        split_docs = text_splitter.split_documents(docs)

        # 캐싱을 지원하는 임베딩 설정
        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
        cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
            embeddings, LocalFileStore(CACHE_DIR), namespace=EMBEDDING_MODEL
        )

        # 벡터 DB 저장
        vectorstore = FAISS.from_documents(split_docs, embedding=cached_embeddings)
        vectorstore.save_local(FAISS_INDEX_DIR)

        return vectorstore
    except Exception as e:
        logger.error(f"Error processing CSV file: {str(e)}")
        logger.error(traceback.format_exc())
        return None

@app.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """CSV 파일 업로드 및 처리"""
    try:
        # 세션 ID 생성
        session_id = str(uuid.uuid4())
        
        # 파일 저장
        file_path = f"./.cache/files/{session_id}_{file.filename}"
        os.makedirs("./.cache/files", exist_ok=True)
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # CSV 파일 처리
        vectorstore = process_csv_file(file_path)
        if vectorstore is None:
            raise HTTPException(status_code=500, detail="Failed to process CSV file")

        # 체인 생성
        chain = create_chain(vectorstore)
        if chain is None:
            raise HTTPException(status_code=500, detail="Failed to create chain")
        
        # 세션에 체인 저장
        chains[session_id] = chain
        
        logger.info(f"File processed successfully with session ID: {session_id}")
        return UploadResponse(session_id=session_id, message="File processed successfully")
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """채팅 요청 처리"""
    try:
        if request.session_id not in chains:
            raise HTTPException(
                status_code=404,
                detail="Session not found. Please upload a file first."
            )
        
        chain = chains[request.session_id]
        logger.info(f"Processing chat request: {request.question}")
        response = chain.invoke(request.question)
        return ChatResponse(answer=response)
    except Exception as e:
        logger.error(f"Error during chat: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

def format_docs(docs):
    return "\n\n".join(
        f"<document><content>{doc.page_content}</content><source>{doc.metadata['source']}</source></document>"
        for doc in docs
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
