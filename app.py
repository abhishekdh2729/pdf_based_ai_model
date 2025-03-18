import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS  # Updated import as per LangChain changes
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Google Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to extract text from PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            extracted_text = page.extract_text()
            if extracted_text:  # Ensure text is not None
                text += extracted_text
    return text

# Function to split text into chunks for vector storage
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

# Function to store text embeddings using FAISS
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")  # Fixed typo from "fiass_index"

# Function to create a conversational chatbot chain
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not available,
    just say "The answer is not available in the context." Do not make up information.

    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """
    
    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3)

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# Function to process user input and fetch response
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Check if FAISS index exists before loading
    if not os.path.exists("faiss_index/index.faiss"):
        st.error("FAISS index not found! Please upload and process PDFs first.")
        return None

    # Load FAISS index
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    # Perform similarity search on stored embeddings
    docs = new_db.similarity_search(user_question)

    # Get chatbot response
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

    return response.get("output_text", "No response generated.")

# Streamlit app interface
def main():
    st.set_page_config(page_title="Chat with PDF")
    st.header("Chat with PDF 💁")

    # User input field for questions
    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        response = user_input(user_question)
        if response:
            st.write("**Answer:**", response)  # Display response

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload PDF Files and click 'Submit & Process'", accept_multiple_files=True)

        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    if raw_text.strip():
                        text_chunks = get_text_chunks(raw_text)
                        get_vector_store(text_chunks)
                        st.success("Processing complete! ✅")
                        st.write("**Extracted Text Preview:**")
                        st.write(raw_text[:500])  # Show first 500 characters
                    else:
                        st.error("No text extracted from PDFs. Please upload valid files.")
            else:
                st.error("Please upload at least one PDF file.")

if __name__ == "__main__":
    main()
