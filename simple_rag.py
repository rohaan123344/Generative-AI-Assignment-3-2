import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq

load_dotenv()

st.set_page_config(page_title="Aicademy 360", layout="wide")

st.title("Aicademy 360 - RAG Simple Portal")
st.write("This is a simple portal for RAG (Retrieval-Augmented Generation) using LangChain and Groq API.")



## api key 

with st.sidebar:
    st.header("Class Settings")
    api_key_input = st.text_input("Enter your Groq API Key", type="password")  



api_key = api_key_input or os.getenv("GROQ_API_KEY")


if not api_key:
    st.warning("Please provide your Groq API Key in the sidebar.")
    st.stop()

os.environ["GROQ_API_KEY"] = api_key

## embedding model

if "embeddings" not in st.session_state:
    st.session_state.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", encode_kwargs={"normalize_embeddings": True})

embeddings = st.session_state.embeddings
# create the LLM

llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0.2)


# Upload documents

uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

col1, col2 = st.columns(2)

with col1:
    chunk_size = st.slider("Chunk Size", min_value=100, max_value=2000, value=500, step=100, help="How much text goes into the chunk")


with col2:
    top_k = st.number_input("Top K", min_value=1, max_value=20, value=5, step=1,help="This is oftem clled Top-K  retrieval")


# Build the knowledge base

if st.button("Build knowledge Base", type="primary"):
    if not uploaded_files:
        st.warning("Please upload at least one PDF file.")
        st.stop()

    all_pages = []

    for uploaded_pdf in uploaded_files:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        temp_file.write(uploaded_pdf.getvalue())
        temp_file.close()

        loader = PyPDFLoader(temp_file.name)
        pages = loader.load()

        for page in pages:
            page.metadata["source_file"] = uploaded_pdf.name
            all_pages.extend(pages)
            os.unlink(temp_file.name)

    #chunking

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=120)

    chunks = text_splitter.split_documents(all_pages)


    # vector database

    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)


    st.session_state.vectorstore = vectorstore
    st.session_state.page_count = len(all_pages)
    st.session_state.chunk_count = len(chunks)
    st.session_state.file_count = len(uploaded_files)
    st.session_state.chunk_size_used = chunk_size

    st.success(f"Knowledge base built successfully! Processed {len(uploaded_files)} files, {len(all_pages)} pages, and created {len(chunks)} chunks.")

if "vectorstore" in st.session_state:
    m1, m2, m3 = st.columns(3)
    m1.metric("Documents", st.session_state.file_count)
    m2.metric("Pages", st.session_state.page_count)
    m3.metric("Chunks", st.session_state.chunk_count)
    st.caption(f"This knowledge base was built with a chunk size of {st.session_state.chunk_size_used} characters.")

    st.divider

    ask_tab, search_tab = st.tabs(["Ask a Question", "Search Knowledge Base"])


    ## TAB 1

    with ask_tab:
        st.subheader("Ask a Question from your documnet")
            
        question = st.text_input("your question", placeholder="Type your question here...",key="rag_question")


        if st.button("Find Evidence and Answer", key="answer_button"):
            if not question.strip():
                st.warning("Please enter a question.")
            else:
                    # Retrieval

                retriever = st.session_state.vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": top_k, "fetch_k": max(10, top_k * 4)},)

                retrieved_docs = retriever.invoke(question)

                context_parts = []

                for number, doc in enumerate(retrieved_docs, start=1):
                    file_name = doc.metadata.get("source_file", "Unknown")
                    page_number = doc.metadata.get("page", 0) + 1

                    context_parts.append(f"Document {number} (File: {file_name}, Page: {page_number}):\n{doc.page_content}\n")

                context = "\n".join(context_parts)

                # Generation

                prompt = f"""
    You are  documnet question answering assistant for students. Answer only from the document evidence below. Use simple language.
    when possible, mention source number like [Source 1]. "If the evidence does not contain the answer, say: "I could not find that answer in the uploaded document."

    DOCUMENT EVIDENCE:
    {context}

    question: {question}"""

                response = llm.invoke(prompt)
                answer = response.content

                st.subheader("AI Answer")
                st.write(answer)

                    # Show the evidence

                st.subheader("Evidence Retrieved from the Document")


                for number, doc in enumerate(retrieved_docs, start=1):
                    file_name = doc.metadata.get("source_file", "Unknown")
                    page_number = doc.metadata.get("page", 0) + 1

                    with st.expander(f"Document {number} (File: {file_name}, Page: {page_number})"):
                        st.write(doc.page_content)

    #Tab 2
    with search_tab:
        st.subheader("Search Knowledge Base")
        st.write("This tab helps us understand that retirval works before the llm writes anything.")

        search_query = st.text_input("Search by meaning", placeholder="Example: model architecture without recurrent network", key="semantic_search")

        if st.button("Search Chunk", placeholder="Example: model architecture without recurrent network", key="search_button"):


            if not search_query.strip():
                st.warning("Please enter a search query.")
            else:
                results = st.session_state.vectorstore.similarity_search(search_query, k=top_k)

                for number, doc in enumerate(results, start=1):
                    file_name = doc.metadata.get("source_file", "Unknown")
                    page_number = doc.metadata.get("page", 0) + 1

                    st.markdown(f"**Document {number} (File: {file_name}, Page: {page_number})**")

                    st.write(doc.page_content)
                    st.divider()

else:
    st.info("Please upload PDF files and build the knowledge base to start asking questions.")
    