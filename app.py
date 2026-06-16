import streamlit as st
import pandas as pd
import plotly.express as px
import os
import json
from PIL import Image
import google.generativeai as genai

# Corrected and Verified LangChain Imports
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.retrieval import create_retrieval_chain # CORRECTED PATH
from langchain.chains.combine_documents import create_stuff_documents_chain # CORRECTED PATH
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

# ==========================================
# 1. SETUP & DUMMY DATA GENERATION
# ==========================================
st.set_page_config(page_title="The Vivriti Oracle", layout="wide")

# This function runs only once to create necessary files
def setup_initial_files():
    if not os.path.exists("dummy_data"):
        os.makedirs("dummy_data")
        # Dummy HR Policy
        with open("dummy_data/HR_Policy.txt", "w") as f:
            f.write("Vivriti Remote Work Policy 2026: Employees in the tech and product teams are allowed 2 days of remote work per week. Mandatory office days are Tuesday and Wednesday. For escalations, contact hr@vivriti.com.")
        # Dummy Cash Flow Data
        cf_data = {
            "Month": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            "Cash_In": [0, 0, 5000, 15000, 25000, 30000, 35000, 40000, 45000, 50000, 50000, 55000],
            "Cash_Out": [-50000, -20000, -10000, -5000, -5000, -5000, -5000, -5000, -5000, -5000, -5000, -5000]
        }
        pd.DataFrame(cf_data).to_csv("dummy_data/Project_CashFlows.csv", index=False)

    if not os.path.exists("tickets.json"):
        with open("tickets.json", "w") as f:
            json.dump([], f)

setup_initial_files()

# ==========================================
# 2. STATE MANAGEMENT & API CONFIG
# ==========================================
with st.sidebar:
    st.title("⚙️ Settings")
    api_key = st.text_input("Enter Gemini API Key", type="password")
    st.info("Get your key at aistudio.google.com")

if not api_key:
    st.warning("Please enter your Gemini API Key in the sidebar to start the Oracle.")
    st.stop()

# Configure APIs
os.environ["GOOGLE_API_KEY"] = api_key
genai.configure(api_key=api_key)

# Initialize Embeddings and LLM
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

@st.cache_resource
def load_knowledge_base():
    if os.path.exists("faiss_index"):
        return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    else:
        documents = []
        for file_name in os.listdir("dummy_data"):
            if file_name.endswith(".txt"):
                with open(os.path.join("dummy_data", file_name), "r") as f:
                    documents.append(Document(page_content=f.read()))
        if not documents:
            documents.append(Document(page_content="This is a placeholder document."))
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(documents)
        vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
        vectorstore.save_local("faiss_index")
        return vectorstore

vector_store = load_knowledge_base()

def load_tickets():
    with open("tickets.json", "r") as f:
        return json.load(f)

def save_tickets(tickets):
    with open("tickets.json", "w") as f:
        json.dump(tickets, f)

# ==========================================
# 3. UI LAYOUT & NAVIGATION
# ==========================================
st.title("The Vivriti Oracle 🔮")
st.markdown("*Institutional Intelligence & Knowledge Synthesis Engine*")

tab1, tab2, tab3 = st.tabs(["💬 Employee Chat", "📊 Finance Visualizer", "👔 CXO Bridge"])

with tab1:
    st.subheader("Ask the Oracle")
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input("Ask about policies or structures"):
        st.chat_message("user").write(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        retriever = vector_store.as_retriever()
        system_prompt = (
            "You are the Vivriti Oracle. Use the provided context to answer the user's question. "
            "If the answer is not in the context, strictly reply with: 'CONFIDENCE_LOW'."
            "\n\nContext:\n{context}"
        )
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])

        question_answer_chain = create_stuff_documents_chain(llm, prompt_template)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)

        response = rag_chain.invoke({"input": prompt})
        answer = response["answer"]

        if "CONFIDENCE_LOW" in answer:
            st.chat_message("assistant").write("I'm sorry, I don't have enough information to answer. Please route this to an expert.")
            with st.form("escalation_form"):
                st.write("### 🚨 Route to Expert")
                dept = st.selectbox("Select Department", ["HR", "Credit", "Capital Markets", "Tech"])
                query = st.text_area("Your Query", value=prompt)
                if st.form_submit_button("Raise CXO Ticket & Show Contact"):
                    tickets = load_tickets()
                    tickets.append({"id": len(tickets)+1, "dept": dept, "query": query, "status": "Open"})
                    save_tickets(tickets)
                    st.success("Ticket raised!")
                    st.info(f"**Contact Info:**\n* Department Head: {dept}_head@vivriti.com\n* Teams: ping {dept} Lead")
        else:
            st.chat_message("assistant").write(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})

with tab2:
    st.subheader("Interactive Cash Flow Visualizer")
    df = pd.read_csv("dummy_data/Project_CashFlows.csv")
    df['Net_Cash_Flow'] = df['Cash_In'] + df['Cash_Out']
    df['Cumulative_Cash'] = df['Net_Cash_Flow'].cumsum()

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("#### Monthly Cash Inflows vs Outflows")
        fig = px.bar(df, x='Month', y=['Cash_In', 'Cash_Out'], barmode='group', color_discrete_sequence=['#2ecc71', '#e74c3c'])
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("#### Cumulative Cash Flow (Breakeven Analysis)")
        fig2 = px.line(df, x='Month', y='Cumulative_Cash', markers=True)
        fig2.add_hline(y=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig2, use_container_width=True)
    with col2:
        st.markdown("#### The Bottom Line 💡")
    
