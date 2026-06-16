import streamlit as st
import pandas as pd
import plotly.express as px
import os
import json
from PIL import Image
import google.generativeai as genai

# ==========================================
# ROBUST LANGCHAIN IMPORTS
# ==========================================
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# ==========================================
# 1. SETUP & DUMMY DATA GENERATION
# ==========================================
st.set_page_config(page_title="The Vivriti Oracle", layout="wide")

# Generate Dummy Data if it doesn't exist
if not os.path.exists("dummy_data"):
    os.makedirs("dummy_data")

    # Dummy HR Policy
    with open("dummy_data/HR_Policy.txt", "w") as f:
        f.write("Vivriti Remote Work Policy 2026: Employees in the tech and product teams are allowed 2 days of remote work per week. Mandatory office days are Tuesday and Wednesday. For escalations, contact hr@vivriti.com.")

    # Dummy Cash Flow Data - SYNTAX CORRECTED
    cf_data = {
        "Month": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        "Cash_In": [0, 0, 5000, 15000, 25000, 30000, 35000, 40000, 45000, 50000, 50000, 55000],
        "Cash_Out": [-50000, -20000, -10000, -5000, -5000, -5000, -5000, -5000, -5000, -5000, -5000, -5000]
    }
    pd.DataFrame(cf_data).to_csv("dummy_data/Project_CashFlows.csv", index=False)

if not os.path.exists("tickets.json"):
    with open("tickets.json", "w") as f:
        json.dump([], f)

# ==========================================
# 2. STATE MANAGEMENT & API CONFIG
# ==========================================
# Sidebar for API Key
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

# Load or Create FAISS Knowledge Base
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
            documents.append(Document(page_content="This is a dummy document to prevent errors on first run."))

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(documents)
        vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
        vectorstore.save_local("faiss_index")
        return vectorstore

vector_store = load_knowledge_base()

# Helper: Load Tickets
def load_tickets():
    with open("tickets.json", "r") as f:
        return json.load(f)

# Helper: Save Tickets
def save_tickets(tickets):
    with open("tickets.json", "w") as f:
        json.dump(tickets, f)

# ==========================================
# 3. UI LAYOUT & NAVIGATION
# ==========================================
st.title("The Vivriti Oracle 🔮")
st.markdown("
