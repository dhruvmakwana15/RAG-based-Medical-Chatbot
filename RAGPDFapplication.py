from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from dotenv import load_dotenv
import streamlit as st
import tempfile


# Load env
path = r"C:\Users\dhruv\OneDrive\Desktop\Langchain_model\Chatmodel\.env"
load_dotenv(path)

# LLM
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task="conversational",
    max_new_tokens=200
)
model = ChatHuggingFace(llm=llm)

# Embedding
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
st.title("Upload PDF and Ask Q&A About your Medical Issues")

uploaded_files = st.file_uploader("Upload a PDF", type="pdf", accept_multiple_files=True)

all_docs = []

if uploaded_files:
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            file_path = tmp_file.name

      # Load PDF\
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        all_docs.extend(docs)
    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=150,
        chunk_overlap=25
    )
    chunks = splitter.split_documents(all_docs)

    # Store in Chroma
    vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embedding,
    collection_name="sample1"
    )

    vector_store.add_documents(chunks)

    st.success(f"Total pages loaded: {len(all_docs)}")
    # Query
    query = st.text_input("Ask a question")
    if query:
        # Retriever
        retriever = vector_store.as_retriever(search_type="similarity",search_kwargs={"k": 2})

        # Prompt — instructs LLM to use context if relevant, else answer from own knowledge
        prompt = PromptTemplate(
            template="""You are a medical assistant AI. Answer the user's question ONLY using the provided context.
Guidelines:
- Provide clear, simple, and accurate explanations
- If the answer is not in the context, say: "I don't know based on the provided document"
- Do NOT make up information
- Do NOT provide medical diagnosis or prescriptions
- If the question is serious, suggest consulting a qualified doctor
- Keep answers concise and easy to understand

Context:
{context}

Question:
{question}

Answer:
""",
            input_variables=["context", "question"]
        )  

        # Retrieve docs
        retrieved_docs = retriever.invoke(query)

        # Combine retrieved content
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])

        # Build final prompt string
        final_prompt = prompt.invoke({
            "context": context,
            "question": query
        })

        # LLM call
        answer = model.invoke(final_prompt)
        st.write("Answer")
        st.write(answer.content)
