"""
DocBuddy — Multi-document RAG Q&A with source citations.

Upload multiple PDFs (lecture notes, policy docs, papers), ask questions
across all of them, and get answers grounded in the actual text — with the
source filename and page number cited for every claim.

A collapsible "Retrieved Context" panel shows exactly which chunks were
pulled from the vector store for the last question, so you can see how
RAG works from the inside.
"""

import os
import shutil

import gradio as gr
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

PERSIST_DIR = "./chroma_store"
COLLECTION_NAME = "docbuddy"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
RETRIEVAL_K = 5


embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


vectorstore = None
if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
    )


def get_index_stats() -> str:
    """Return a human-readable 'X documents indexed — Y total chunks' string."""
    if vectorstore is None:
        return "No documents indexed yet."

    data = vectorstore.get()
    metadatas = data.get("metadatas", []) or []
    total_chunks = len(metadatas)

    if total_chunks == 0:
        return "No documents indexed yet."

    sources = {m.get("source", "unknown") for m in metadatas}
    return f"{len(sources)} document(s) indexed — {total_chunks} total chunks"



def index_documents(pdf_paths: list) -> int:
    """
    Load each PDF, split into chunks, tag with source/page metadata,
    embed, and persist to a Chroma vector store.

    Returns the total number of chunks added.
    """
    global vectorstore

    if not pdf_paths:
        return 0

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    all_chunks = []
    for pdf_path in pdf_paths:
        filename = os.path.basename(pdf_path)
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()

        chunks = splitter.split_documents(pages)

        for chunk in chunks:
            raw_page = chunk.metadata.get("page", 0)
            chunk.metadata["source"] = filename
            chunk.metadata["page"] = raw_page + 1

        all_chunks.extend(chunks)
        print(f"[index_documents] {filename}: {len(pages)} page(s) -> {len(chunks)} chunk(s)")

    if vectorstore is None:
        vectorstore = Chroma.from_documents(
            documents=all_chunks,
            embedding=embeddings,
            collection_name=COLLECTION_NAME,
            persist_directory=PERSIST_DIR,
        )
    else:
        vectorstore.add_documents(all_chunks)

    total_chunks = len(all_chunks)
    print(f"[index_documents] Indexed {total_chunks} new chunk(s) from {len(pdf_paths)} file(s).")
    return total_chunks



GROUNDED_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are DocBuddy, a careful research assistant. Answer the "
            "user's question using ONLY the information in the context "
            "below, which was retrieved from the user's uploaded documents.\n\n"
            "Rules:\n"
            "1. If the context does not contain enough information to answer "
            "the question, respond exactly with: \"I don't have that "
            "information in the provided documents.\" Do not guess, and do "
            "not use any outside knowledge.\n"
            "2. When you do answer, cite the source document and page number "
            "for every claim, in the format (Source: <filename>, Page: <page>).\n"
            "3. If the context comes from multiple documents, synthesize "
            "across them and cite each source you used.\n"
            "4. Be concise and accurate. Do not invent page numbers, "
            "filenames, or facts that are not in the context.\n\n"
            "Context:\n{context}",
        ),
        ("human", "{question}"),
    ]
)

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)


def _format_context_for_prompt(docs) -> str:
    """Build the {context} string injected into the grounded prompt."""
    blocks = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        blocks.append(f"[Source: {source}, Page: {page}]\n{doc.page_content}")
    return "\n\n---\n\n".join(blocks)


def _format_context_display(docs) -> str:
    """Build the markdown shown in the 'Retrieved Context' accordion."""
    if not docs:
        return "_No chunks retrieved._"

    lines = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        preview = doc.page_content.strip().replace("\n", " ")
        if len(preview) > 300:
            preview = preview[:300] + "..."
        lines.append(
            f"**Chunk {i}** — `{source}`, page **{page}**\n\n"
            f"> {preview}\n"
        )
    return "\n---\n".join(lines)


def ask(question: str) -> tuple:
    """
    Retrieve the top-k chunks for `question`, ask the LLM to answer using
    only that context, and return (answer, context_display_markdown).
    """
    if vectorstore is None:
        return (
            "No documents have been indexed yet. Upload one or more PDFs "
            "and click 'Index Documents' first.",
            "_No chunks retrieved — no documents indexed yet._",
        )

    if not question or not question.strip():
        return "Please enter a question.", "_No chunks retrieved._"

    retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVAL_K})
    docs = retriever.invoke(question)

    context_text = _format_context_for_prompt(docs)
    context_display = _format_context_display(docs)

    chain = GROUNDED_PROMPT | llm
    response = chain.invoke({"context": context_text, "question": question})

    return response.content, context_display


def handle_index(files):
    """Callback for the 'Index Documents' button."""
    if not files:
        return get_index_stats()

    pdf_paths = [f.name if hasattr(f, "name") else f for f in files]
    new_chunks = index_documents(pdf_paths)
    stats = get_index_stats()

    if new_chunks == 0:
        return stats
    return stats


def handle_ask(question, history):
    """Callback for submitting a question in the chat box."""
    history = history or []

    if not question or not question.strip():
        return history, "", "_No chunks retrieved._"

    answer, context_display = ask(question)

    history = history + [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]
    return history, "", context_display


with gr.Blocks(title="DocBuddy — Multi-Document RAG") as demo:
    gr.Markdown(
        "# 📚 DocBuddy\n"
        "Upload PDFs (lecture notes, policy documents, research papers), "
        "then ask questions across all of them. Every answer cites the "
        "source document and page number it came from."
    )

    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(
                label="Upload PDFs",
                file_count="multiple",
                file_types=[".pdf"],
            )
            index_button = gr.Button("📥 Index Documents", variant="primary")
            status_label = gr.Label(value=get_index_stats(), label="Index Status")

        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="DocBuddy", height=400)
            question_input = gr.Textbox(
                label="Ask a question",
                placeholder="e.g. What does the policy say about remote work eligibility?",
                lines=1,
            )
            ask_button = gr.Button("🔎 Ask", variant="primary")

    with gr.Accordion("🔍 Retrieved Context", open=False):
        context_display = gr.Markdown("_No chunks retrieved yet — ask a question to see what RAG retrieves._")

    # Wiring
    index_button.click(
        fn=handle_index,
        inputs=[file_input],
        outputs=[status_label],
    )

    ask_button.click(
        fn=handle_ask,
        inputs=[question_input, chatbot],
        outputs=[chatbot, question_input, context_display],
    )

    question_input.submit(
        fn=handle_ask,
        inputs=[question_input, chatbot],
        outputs=[chatbot, question_input, context_display],
    )


if __name__ == "__main__":
    demo.launch()