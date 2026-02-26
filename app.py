import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains.retrieval_qa.base import RetrievalQA

# Page Setup
st.set_page_config(
    page_title="Medical RAG Assistant", 
    page_icon="🏥", 
    layout="wide"
)

# Load environment variables
load_dotenv()

# Custom CSS for better UI with fixed contrast
st.markdown("""
    <style>
    .main-title {
        text-align: center;
        color: #4FC3F7;
        font-size: 3rem;
        margin-bottom: 0;
        font-weight: 600;
    }
    .subtitle {
        text-align: center;
        color: #B0BEC5;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    /* Sidebar styling - 밝은 흰 톤 */
    [data-testid="stSidebar"] {
        background-color: #F8F9FA;
        border-right: 1px solid #E0E0E0;
    }
    
    /* 사이드바 내 모든 텍스트 요소를 어두운 색으로 */
    [data-testid="stSidebar"],
    [data-testid="stSidebar"] * {
        color: #333333;
    }
    
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4,
    [data-testid="stSidebar"] h5,
    [data-testid="stSidebar"] h6 {
        color: #1A1A1A !important;
    }
    
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] li,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] div {
        color: #333333 !important;
    }
    
    /* 입력 필드 텍스트 */
    [data-testid="stSidebar"] input {
        color: #1A1A1A !important;
        background-color: #FFFFFF !important;
        border-color: #CCCCCC !important;
    }
    
    /* 도움말(help) 텍스트 */
    [data-testid="stSidebar"] .stTooltipIcon {
        color: #888888 !important;
    }
    
    /* 구분선 */
    [data-testid="stSidebar"] hr {
        border-color: #DDDDDD;
    }
    
    /* Chat message styling */
    .stChatMessage {
        background-color: #2C2C2C;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    
    /* User message */
    [data-testid="stChatMessageContent"] {
        color: #FFFFFF;
    }
    
    /* Source box styling with better contrast */
    .source-box {
        background-color: #3D3D3D;
        border-left: 4px solid #FFC107;
        padding: 1rem;
        margin-top: 1rem;
        border-radius: 5px;
        font-size: 0.95rem;
        color: #E0E0E0;
    }
    
    .source-box strong {
        color: #FFC107;
    }
    
    /* Code block in sidebar */
    [data-testid="stSidebar"] code {
        color: #4FC3F7 !important;
        background-color: #2C2C2C !important;
    }
    
    /* Status boxes */
    .stSuccess, .stWarning, .stInfo {
        color: #1E1E1E !important;
    }
    
    /* Footer styling */
    .footer-text {
        text-align: center;
        color: #90A4AE;
        font-size: 0.85rem;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-title">🏥 Medical RAG Assistant</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Ask questions about medical transcriptions powered by AI</p>', unsafe_allow_html=True)
st.markdown("---")

# Minimal Sidebar - Just API Key Status
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Check if there's an API key in the environment
    env_api_key = os.getenv("OPENAI_API_KEY", "")
    
    # Input field for API Key
    api_key_input = st.text_input(
        "OpenAI API Key", 
        type="password", 
        value=env_api_key,
        help="Paste your OpenAI API key here to start using the assistant.",
        key="api_key_input"
    )
    
    if api_key_input:
        st.success("✅ OpenAI API Key Loaded")
        st.info("💡 System Ready")
    else:
        st.warning("⚠️ Please provide an OpenAI API Key")
    
    st.markdown("---")
    st.markdown("### 📝 Sample Questions")
    st.markdown("""
    <div style="color: #555555;">
    • What are the symptoms of allergic rhinitis?<br>
    • Describe the procedure for laparoscopic gastric bypass<br>
    • What is the treatment for chronic sinusitis?<br>
    • How is a lumbar puncture performed?<br>
    • What are the signs of acute appendicitis?
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 🔧 Configuration")
    st.code("""Model: gemini-2.5-flash
Temperature: 0.3
Retrieval: Top 3 documents
Embeddings: text-embedding-004""", language="yaml")

# Load RAG Chain with key arguments passed directly
@st.cache_resource
def load_chain(api_key):
    """Load the RAG chain with user-provided API key"""
    
    # Check if vectorstore exists
    if not os.path.exists("vectorstore"):
        return None, "❌ 'vectorstore' folder not found. Please ensure it exists in the directory."
    
    try:
        # Initialize embeddings with the provided key
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        
        # Load vector store
        vector_store = FAISS.load_local(
            "vectorstore", 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        
        # Create retriever with k=3 (top 3 most relevant documents)
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        
        # Initialize LLM with the provided key
        llm = ChatOpenAI(
            model="gpt-3.5-turbo", 
            temperature=0.3,
            max_tokens=1024,
            openai_api_key=api_key
        )
        # Define prompt template
        custom_prompt_template = """
You are a knowledgeable medical assistant. Use the following retrieved context to answer the question.
Always respond in the same language as the user's question (e.g., if the question is in Korean, answer in Korean).

Rules:
- Be concise and clear. Do NOT repeat words or phrases.
- Only list each symptom, finding, or fact ONCE.
- If the answer is not in the context, say so clearly.
- Cite the specialty or source for key facts.

Context:
{context}

Question: {question}

Answer:
"""
        
        prompt = PromptTemplate(
            template=custom_prompt_template, 
            input_variables=["context", "question"]
        )
        
        # Build the QA chain
        chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )
        
        return chain, "✅ System loaded successfully"
        
    except Exception as e:
        return None, f"❌ Error loading system: {str(e)}"

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=True)

# Chat input
if query := st.chat_input("💬 Ask a medical question..."):
    
    # Check for API key
    if not api_key_input:
        st.error("⚠️ Please set your OpenAI API Key in the sidebar first.")
        st.stop()
    
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)
    
    # Generate assistant response
    with st.chat_message("assistant"):
        # Load chain with the provided API key
        chain, status = load_chain(api_key=api_key_input)
        
        if chain:
            with st.spinner("🔍 Searching medical records..."):
                try:
                    # Get response from chain
                    response = chain.invoke({"query": query})
                    answer = response['result']
                    
                    # Extract and format sources
                    sources = []
                    for doc in response['source_documents']:
                        source_name = doc.metadata.get('source', 'Unknown')
                        specialty = doc.metadata.get('specialty', 'General')
                        source_entry = f"<strong>{source_name}</strong> <em>({specialty})</em>"
                        if source_entry not in sources:
                            sources.append(source_entry)
                    
                    # Display answer first
                    st.markdown(answer)
                    
                    # Display sources in a styled box
                    if sources:
                        sources_html = "<br>".join([f"• {s}" for s in sources])
                        st.markdown(
                            f'<div class="source-box"><strong>📚 Sources Used:</strong><br><br>{sources_html}</div>',
                            unsafe_allow_html=True
                        )
                    
                    # Format full response for history
                    sources_text = "\n".join([f"• {s}" for s in sources])
                    full_response = f"{answer}\n\n**📚 Sources Used:**\n{sources_text}"
                    
                    # Save to chat history
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer + f'\n\n<div class="source-box"><strong>📚 Sources Used:</strong><br><br>{sources_html}</div>' if sources else answer
                    })
                    
                except Exception as e:
                    error_msg = f"❌ Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": error_msg
                    })
        else:
            # System not loaded properly
            st.error(status)
            st.info("💡 Make sure the 'vectorstore' folder with index.faiss and index.pkl exists in your directory.")

# Footer
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown(
        '<p class="footer-text">'
        '🏥 Medical RAG Assistant | Powered by LangChain & Google Gemini'
        '</p>', 
        unsafe_allow_html=True
    )