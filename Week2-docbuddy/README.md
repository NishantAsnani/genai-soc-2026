# DocBuddy

DocBuddy is a small RAG-powered app that lets you upload PDF documents, index them into a vector store, and ask questions across all of the documents at once. It returns answers grounded in the actual text, with inline citations showing the source filename and page number.

## Screenshots

![DocBuddy Upload Interface](Screenshot%202026-06-14%20115637.png)

![DocBuddy Indexing Status](Screenshot%202026-06-14%20115757.png)

![DocBuddy Answer with Citations](Screenshot%202026-06-14%20115955.png)

These images show the full workflow: uploading PDFs, indexing them into the vector store, asking a question, and receiving an answer with citations that connect back to the source documents.

## What is RAG?

RAG stands for Retrieval-Augmented Generation. It means the app first retrieves relevant passages from uploaded documents, then uses those passages to generate an answer, so the response is grounded in actual source material instead of made up from scratch. In this project, RAG helps DocBuddy answer questions using the exact text from PDFs and cite where the information came from.

## Install steps

1. Clone the repository or open the `week2-docbuddy/` folder.
2. Create a Python virtual environment:
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```
3. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
4. Set your Groq API key in a `.env` file or environment variable:
   ```powershell
   GROQ_API_KEY="your_api_key_here"
   ```
5. Run the app:
   ```powershell
   python app1.py
   ```
6. Open the Gradio URL shown in the terminal and upload one or more PDFs, then click **Index Documents** and ask a question.

## What worked well

- Uploading PDFs and splitting them into manageable chunks worked reliably.
- The app indexes documents into a Chroma vector store so later queries are fast.
- Answers are generated with citations for each claim, which makes the results much more trustworthy.
- The retrieved context panel helps show exactly which document chunks were used for the answer.

## What I would improve

- Add support for more file types like Word documents and text files.
- Improve the UI to show citations directly in the chat response, not just in the prompt instructions.
- Add a progress indicator for indexing large PDFs.
- Make the vector store update more smoothly when documents are re-uploaded or deleted.

## Notes

This README is written to explain what I built and what I learned while building it, without including any testing instructions.
