#  🩺 Medical AI Chatbot (RAG + PDF + Streamlit)

An AI-powered medical assistant that allows users to upload multiple PDF documents and ask questions based on their content.
Built using LangChain, HuggingFace LLaMA-3, Chroma Vector DB, and Streamlit for an interactive experience.  
Built by Dhruv Makwana

## 🚀 Features
📄 Upload multiple PDF files  
🧠 Retrieval-Augmented Generation (RAG) for accurate answers  
🔍 Semantic search using embeddings  
🤖 Context-aware responses using LLaMA-3  
⚡ Real-time Q&A with Streamlit UI  
🛡️ Reduces hallucination using strict prompt design  

## 🛠️ Tech Stack
- Python
- LangChain
- HuggingFace (LLaMA-3 + Embeddings)
- Chroma (Vector Database)
- Streamlit
- dotenv

## ⚙️ Setup Instructions
 1. Clone Repository  
git clone https://github.com/your-username/your-repo-name.git  
cd your-repo-name  
 2. Create Virtual Environment  
python -m venv venv  
venv\Scripts\activate    
 3. Install Dependencies  
pip install -r requirements.txt  
 4. Add API Key  
Create .env file:  
HUGGINGFACEHUB_API_TOKEN=your_api_key_here  
 5. Run the Application  
streamlit run app.py

## 💡 How It Works
📄 User uploads one or more PDF files  
✂️ Documents are split into smaller chunks  
🔢 Chunks are converted into embeddings (vector format)  
🗄️ Stored in Chroma vector database  
🔍 User query → similarity search retrieves relevant chunks  
🤖 LLM (LLaMA-3) generates answer based on retrieved context  

## 📸 Screenshots
<img width="1428" height="963" alt="image" src="https://github.com/user-attachments/assets/dbdd238f-a9c7-40df-80ee-cef14cd25fed" />
