from dotenv import load_dotenv
import os
from PyPDF2 import PdfReader
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms import OpenAI
from langchain_community.callbacks import get_openai_callback

# Load environment variables
load_dotenv()

def extract_text_from_pdf(pdf_path):
    text = ''
    with open(pdf_path, 'rb') as file:
        reader = PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text

def process_text(text):
    # Split the text into chunks using langchain
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    
    # Convert the chunks of text into embeddings to form a knowledge base
    embeddings = OpenAIEmbeddings()
    knowledgeBase = FAISS.from_texts(chunks, embeddings)
    
    return knowledgeBase

def main(filename, query):
    # Text variable will store the pdf text
    print("Extracting text from pdf...")
    text = extract_text_from_pdf(filename)
    
    # Create the knowledge base object
    print("Creating knowledge base...")
    knowledgeBase = process_text(text)
    
    print("Searching for data...")
    docs = knowledgeBase.similarity_search(query)
    llm = OpenAI()
    chain = load_qa_chain(llm, chain_type='stuff')
        
    print("Running query...")
    with get_openai_callback() as cost:
        response = chain.run(input_documents=docs, question=query)
        print("\n\nResponse:\n", response)
        print("\n\nCost:\n", cost)
                
                  
main("source.pdf", "What is a neural style transfer?")