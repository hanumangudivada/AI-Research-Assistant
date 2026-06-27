from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.chat_history import (
    InMemoryChatMessageHistory,
    BaseChatMessageHistory,
)
from langchain_core.runnables.history import RunnableWithMessageHistory
from typing import List, Dict, Optional
from langchain_core.documents import Document

from src.research_response import ResearchResponse
from langchain_community.document_loaders import TextLoader

from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda

import shutil

shutil.rmtree("./research_db", ignore_errors=True)

load_dotenv()

class AIResearchAssistant:
    """AI Research Assistant with document ingestion and retrieval."""
    def __init__(
        self,
         persist_directory: str= "./research_db",
        chunk_size:int=300,
        chunk_overlap:int=100):
        
        self.persist_directory = persist_directory

        self.embeddings=HuggingFaceEmbeddings(
           model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        self.llm=init_chat_model(model="llama-3.3-70b-versatile",model_provider="groq",
         temperature=0)
        
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        self.vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings,
            collection_name="research_docs",
        )
        
        self.session_store: Dict[str,InMemoryChatMessageHistory]={}
        self.stuctured_llm= self.llm.with_structured_output(ResearchResponse)
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are an AI Research Assistant. Analyze the provided documents 
    and return a structured response.
     ONLY answer using information directly related to the user's question.

    Do not include unrelated facts even if they appear in the retrieved context.

    Rules:
    1. ONLY use information from the provided context
    2. If the context doesn't have the answer, say so in the answer field
    3. Set confidence: "high" if directly stated, "medium" if inferred, "low" if partial
    5. Extract key quotes word-for-word from the context
    6. Suggest 2-3 follow-up questions the user might want to ask

     Use conversation history to understand follow-up questions.""",
                ),
                MessagesPlaceholder(variable_name="history"),
                (
                    "human",
                    """Context documents:

    {context}

    Question: {question}""",
                ),
            ]
        )
    

        self.rag_chain = (
         RunnableParallel(
        context=RunnableLambda(
            lambda x: self.format_docs_for_context(
                self.retriver(x["question"])
            )
        ),
        question=RunnableLambda(
            lambda x: x["question"]
        ),
        history=RunnableLambda(
            lambda x: x["history"]
        ),
    )
    | self.prompt
    | self.stuctured_llm
)
        
    

    def load_document(self,file_path: str):
        #Load the text file
        loader = TextLoader(file_path)
        documents= loader.load()
        return documents

    def add_documents(self,
            documents:List[Document],
            ):
         
         # Split into chunks
        chunks = self.splitter.split_documents(documents)
        self.vectorstore.add_documents(chunks)
        print(chunks)
        return len(chunks)
    
    
    def retriver(self, query:str):
        Multi_retriver= MultiQueryRetriever.from_llm(retriever=self.vectorstore.as_retriever(search_kwargs={"k": 5}),llm=self.llm)
        Compressor= LLMChainExtractor.from_llm(self.llm)

        Enhanced_retriver= ContextualCompressionRetriever(
            base_compressor=Compressor,
            base_retriever=Multi_retriver
        )

        docs= Enhanced_retriver.invoke(query)
        print(docs)
        return docs
    
    def format_docs_for_context(self,docs):
        """Format retrieved documents into a string for the prompt."""
        if not docs:
            return "No relevant documents found."
        
        formatted=[]
        for i, doc in enumerate(docs):
            formatted.append(f"{doc.page_content}")
        return "\n\n---\n\n".join(formatted)

    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        """Get or create session history."""
        if session_id not in self.session_store:
            self.session_store[session_id] = InMemoryChatMessageHistory()
        return self.session_store[session_id]
    


        
    
    def research(self, query:str,session_id:str):
        history = self.get_session_history(session_id)
        response=self.rag_chain.invoke({"question": query,"history": history.messages})
        history.add_user_message(query)
        history.add_ai_message(response.answer)
        print(history)
        print("***********************")
        print(response)
        return response

    
    
    
    
               





