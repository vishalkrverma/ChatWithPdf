"""
ChatPDF with Gemini AI - Single File App

Built using:
- Streamlit for UI
- PyPDF2 for PDF extraction
- LangChain for prompt handling and chaining
- FAISS for vector similarity search
- Google Generative AI (Gemini) for Q&A, summary, and challenge question generation

How to Run:
1. Make sure you have Python 3.9+ and pip installed
2. Create and activate virtual environment (optional but recommended)
3. Install dependencies: pip install -r requirements.txt
4. Create a `.env` file with your API key:
   GOOGLE_API_KEY=your_gemini_api_key_here
5. Run this file:
   streamlit run chatpdf_app.py
"""