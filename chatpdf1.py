import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load and verify API key
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error("❌ GOOGLE_API_KEY not found. Please check your .env file.")
    st.stop()

genai.configure(api_key=api_key)

# Utility Functions
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return splitter.split_text(text)

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    store = FAISS.from_texts(text_chunks, embedding=embeddings)
    store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context.
    If the answer is not in the context, say "answer is not available in the context".

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    try:
        db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        st.error(f"Error loading FAISS index: {e}")
        return
    docs = db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("💬 Reply:", response["output_text"])

def generate_summary(text):
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
    prompt = f"Summarize the following document in no more than 150 words:\n\n{text[:8000]}"
    return model.invoke(prompt)

def generate_challenge_questions():
    prompt = """
    Based on the following document context, generate 3 challenging, logic-based or comprehension-focused questions.
    Only return the questions in a numbered list.

    Context:
    {context}
    """
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
    prompt_template = PromptTemplate(template=prompt, input_variables=["context"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt_template)

    db = FAISS.load_local("faiss_index", GoogleGenerativeAIEmbeddings(model="models/embedding-001"), allow_dangerous_deserialization=True)
    docs = db.similarity_search("overview of the document")
    result = chain({"input_documents": docs, "question": "Generate 3 questions"}, return_only_outputs=True)

    questions = result["output_text"].split("\n")
    return [q.strip() for q in questions if q.strip() and q[0].isdigit()]

# Main UI
def main():
    st.set_page_config(page_title="Chat PDF App")
    st.header("📄 Chat with PDF using EZ-ChatBot")

    user_question = st.text_input("Ask a question from the PDF files:")
    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("📂 Menu:")
        pdf_docs = st.file_uploader("Upload your PDF files", accept_multiple_files=True)

        if st.button("Submit & Process"):
            with st.spinner("🔄 Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.session_state["summary"] = generate_summary(raw_text)
                st.success("✅ Processing complete!")

    if "summary" in st.session_state:
        st.subheader("📝 Auto Summary")
        st.write(st.session_state["summary"].content)

    st.markdown("---")
    if st.button("🚀 Challenge Me"):
        st.session_state["challenge_qs"] = generate_challenge_questions()
        st.session_state["user_answers"] = [""] * len(st.session_state["challenge_qs"])

    if "challenge_qs" in st.session_state:
        st.subheader("🧠 Challenge Questions")
        for i, q in enumerate(st.session_state["challenge_qs"]):
            st.text(f"Q{i+1}: {q}")
            st.session_state["user_answers"][i] = st.text_input(f"Your Answer to Q{i+1}:", key=f"ans_{i}")

        if st.button("✅ Submit Answers"):
            for i, (q, user_ans) in enumerate(zip(st.session_state["challenge_qs"], st.session_state["user_answers"])):
                full_prompt = f"Evaluate this answer: '{user_ans}' to the question: '{q}' based on the document. Justify your evaluation."
                st.markdown(f"**Feedback for Q{i+1}:**")
                user_input(full_prompt)

if __name__ == "__main__":
    main()
