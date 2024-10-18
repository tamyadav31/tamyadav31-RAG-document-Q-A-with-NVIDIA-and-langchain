import streamlit as st
import os
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings , ChatNVIDIA
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv
load_dotenv()


##load nvidia api key

os.environ['NVIDIA_API_KEY']=os.getenv('NVIDIA_API_KEY')

llm=ChatNVIDIA(model_name="meta/llama3-70b-instruct")

def vector_embedding():
    if 'vectors' not in st.session_state:
        st.session_state.embeddings= NVIDIAEmbeddings()
        st.session_state.loader=PyPDFDirectoryLoader("./us_census")
        st.session_state.docs=st.session_state.loader.load()
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=700 , chunk_overlap=50)
        st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs[:30])
        st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents , st.session_state.embeddings)
        
        
        
st.title("NVIDIA NIM Demo")

prompt=ChatPromptTemplate.from_template(
    """
    answer the questions based on the provided context only 
    please provide the most accurate response based on the question
    <context>
    {context}
    <context>
    questions:{input}
    
    """
)


prompt1=st.text_input("enter your question from documents ")

if st.button("document embedding"):
    vector_embedding()
    st.write("vector store db is ready using nvidia embeddings")
    
import time

if prompt1:
    documents_chain=create_stuff_documents_chain(llm , prompt)
    retriever=st.session_state.vectors.as_retriever()
    retriever_chain=create_retrieval_chain(retriever , documents_chain)
    start=time.process_time()
    response=retriever_chain.invoke({'input':prompt1})
    print("response time :" , time.process_time()-start)
    st.write(response['answer'])
    
    #with a streamlit expander
    with st.expander("document similarity search"):
        #find relevant chats
        for i , doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("-----------------------")


        