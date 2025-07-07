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
import re

# Load API Key
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error("❌ GOOGLE_API_KEY not found. Please check your .env file.")
    st.stop()

genai.configure(api_key=api_key)

# ========== Utility Functions ==========

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

# ========== MCQ Challenge Mode ==========

def generate_mcq_challenge():
    prompt = """
    Based on the following document context, generate 3 multiple-choice questions.
    Each question should have 4 options labeled (A), (B), (C), and (D), and indicate the correct answer with a justification.

    Format:
    1. Question?
       (A) Option A
       (B) Option B
       (C) Option C
       (D) Option D
       Answer: B
       Explanation: Because...

    Context:
    {context}
    """
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
    prompt_template = PromptTemplate(template=prompt, input_variables=["context"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt_template)

    db = FAISS.load_local("faiss_index", GoogleGenerativeAIEmbeddings(model="models/embedding-001"), allow_dangerous_deserialization=True)
    docs = db.similarity_search("overview of the document")
    result = chain({"input_documents": docs, "question": "Generate 3 MCQs"}, return_only_outputs=True)
    st.write(result)
    return parse_mcq_output(result["output_text"])
    # return parse_mcq_output(result.output_text)
def parse_mcq_output(output_text):
    questions = []
    print(output_text)
    blocks = re.split(r"\n\s*\d+\.", "\n" + output_text)
    for block in blocks[1:]:
        q = {}
        lines = block.strip().split("\n")
        q["question"] = lines[0]
        q["options"] = {}
        for line in lines[1:5]:
            match = re.match(r"\((.)\)\s*(.*)", line)
            if match:
                q["options"][match.group(1)] = match.group(2)
        answer_line = next((l for l in lines if l.lower().startswith("answer:")), "")
        q["answer"] = answer_line.split(":")[-1].strip().upper()
        explanation_line = next((l for l in lines if l.lower().startswith("explanation:")), "")
        q["explanation"] = explanation_line.split(":", 1)[-1].strip()
        questions.append(q)
        
    return questions

# ========== Streamlit UI ==========

def main():
    st.set_page_config(page_title="Chat PDF App")
    st.header("📄 Chat with PDF using EZ ChatBot")

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
        st.session_state["challenge_mcqs"] = generate_mcq_challenge()
        st.session_state["mcq_user_choices"] = [""] * 3

    if "challenge_mcqs" in st.session_state:
        st.subheader("🧠 MCQ Challenge")
        for i, mcq in enumerate(st.session_state["challenge_mcqs"]):
            st.markdown(f"**Q{i+1}: {mcq['question']}**")
            options = list(mcq["options"].items())
            selected = st.radio(
                label="Choose an option:",
                options=[f"({k}) {v}" for k, v in options],
                key=f"mcq_{i}"
            )
            if selected:
                st.session_state["mcq_user_choices"][i] = selected.split(")")[0][1]

        if st.button("✅ Submit Answers"):
            for i, mcq in enumerate(st.session_state["challenge_mcqs"]):
                user_ans = st.session_state["mcq_user_choices"][i]
                correct = mcq["answer"]

                st.markdown(f"**Q{i+1}: {mcq['question']}**")
                st.markdown(f"Your Answer: **({user_ans}) {mcq['options'].get(user_ans, '')}**")
                st.markdown(f"Correct Answer: **({correct}) {mcq['options'].get(correct, '')}**")

                if user_ans == correct:
                    st.success("✅ Correct!")
                else:
                    st.error("❌ Incorrect.")
                st.info(f"**Explanation:** {mcq['explanation']}")

if __name__ == "__main__":
    main()
