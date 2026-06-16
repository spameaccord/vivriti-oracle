import streamlit as st
import pandas as pd
import plotly.express as px
import os
import json
from PIL import Image
import google.generativeai as genai

# ==========================================
# ROBUST, SELF-HEALING LANGCHAIN IMPORTS
# ==========================================
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

# This block tries modern and legacy import paths to prevent ModuleNotFoundError
try:
    # The most common, modern path
    from langchain.chains import create_retrieval_chain
except ImportError:
    try:
        # An older, but still possible path
        from langchain.chains.retrieval import create_retrieval_chain
    except ImportError:
        st.error("FATAL ERROR: Could not import 'create_retrieval_chain'. Deployment failed. Check LangChain version.")
        st.stop()

try:
    # The most common, modern path
    from langchain.chains.combine_documents import create_stuff_documents_chain
except ImportError:
    try:
        # An older path
        from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
    except ImportError:
        st.error("FATAL ERROR: Could not import 'create_stuff_documents_chain'. Deployment failed. Check LangChain version.")
        st.stop()

# ==========================================
# 1. SETUP & DUMMY DATA GENERATION
# ==========================================
st.set_page_config(page_title="The Vivriti Oracle", layout="wide")

def setup_initial_files():
    if not os.path.exists("dummy_data"):
        os.makedirs("dummy_data")
        with open("dummy_data/HR_Policy.txt", "w") as f:
            f.write("Vivriti Remote Work Policy 2026: Employees in the tech and product teams are allowed 2 days of remote work per week. Mandatory office days are Tuesday and Wednesday. For escalations, contact hr@vivriti.com.")
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
    st.info("Get your key from Google AI Studio.")

if not api_key:
    st.warning("Please enter your Gemini API Key in the sidebar to start.")
    st.stop()

genai.configure(api_key=api_key)

# ==========================================
# 3. CORE APPLICATION LOGIC
# ==========================================
@st.cache_resource
def get_llm_and_vectorstore(_api_key):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    if os.path.exists("faiss_index"):
        vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    else:
        documents = []
        with open("dummy_data/HR_Policy.txt", "r") as f:
            documents.append(Document(page_content=f.read()))
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(documents)
        vector_store = FAISS.from_documents(documents=splits, embedding=embeddings)
        vector_store.save_local("faiss_index")
    return llm, vector_store

llm, vector_store = get_llm_and_vectorstore(api_key)

def load_tickets():
    if os.path.exists("tickets.json"):
        with open("tickets.json", "r") as f:
            return json.load(f)
    return []

def save_tickets(tickets):
    with open("tickets.json", "w") as f:
        json.dump(tickets, f)

st.title("The Vivriti Oracle 🔮")
st.markdown("*Institutional Intelligence & Knowledge Synthesis Engine*")

tab1, tab2, tab3 = st.tabs(["💬 Employee Chat", "📊 Finance Visualizer", "👔 CXO Bridge"])

with tab1:
    st.subheader("Ask the Oracle")
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])
    if prompt := st.chat_input("Ask about policies or company knowledge..."):
        st.chat_message("user").write(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        retriever = vector_store.as_retriever()
        system_prompt = "You are the Vivriti Oracle. Use provided context to answer. If the answer is not in the context, strictly reply: 'CONFIDENCE_LOW'.\n\nContext:\n{context}"
        prompt_template = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])
        qa_chain = create_stuff_documents_chain(llm, prompt_template)
        rag_chain = create_retrieval_chain(retriever, qa_chain)
        response = rag_chain.invoke({"input": prompt})
        answer = response["answer"]
        if "CONFIDENCE_LOW" in answer:
            st.chat_message("assistant").write("I don't have enough information to answer. Please escalate to an expert.")
            with st.form("escalation_form"):
                st.write("### 🚨 Route to Expert")
                dept = st.selectbox("Select Department", ["HR", "Credit", "Capital Markets", "Tech"])
                if st.form_submit_button("Raise Ticket & Show Contact"):
                    tickets = load_tickets()
                    tickets.append({"id": len(tickets)+1, "dept": dept, "query": prompt, "status": "Open"})
                    save_tickets(tickets)
                    st.success("Ticket raised!")
                    st.info(f"**Contact Info:**\n* Email: {dept}_head@vivriti.com\n* Teams: @{dept} Lead")
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
        st.markdown("##### Monthly Cash Inflows vs Outflows")
        fig = px.bar(df, x='Month', y=['Cash_In', 'Cash_Out'], barmode='group', color_discrete_sequence=['#2ecc71', '#e74c3c'])
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("##### Cumulative Cash Flow (Breakeven Analysis)")
        fig2 = px.line(df, x='Month', y='Cumulative_Cash', markers=True)
        fig2.add_hline(y=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig2, use_container_width=True)
    with col2:
        st.markdown("##### The Bottom Line 💡")
        breakeven_month = df[df['Cumulative_Cash'] > 0]['Month'].min()
        st.metric("Breakeven Point", f"Month {breakeven_month}" if pd.notna(breakeven_month) else "Not Reached")
        st.markdown("##### Raw Data")
        st.dataframe(df, hide_index=True)

with tab3:
    st.subheader("CXO Dashboard: Resolve & Teach")
    tickets = load_tickets()
    open_tickets = [t for t in tickets if t["status"] == "Open"]
    if not open_tickets:
        st.success("No open escalations.")
    else:
        for t in open_tickets:
            with st.expander(f"Ticket #{t['id']} | Dept: {t['dept']} | Query: {t['query']}"):
                resolution_text = st.text_area("Resolution Notes", key=f"res_{t['id']}")
                uploaded_image = st.file_uploader("Upload screenshot", type=["png", "jpg"], key=f"img_{t['id']}")
                if st.button("Close Ticket & Update Oracle", key=f"btn_{t['id']}"):
                    final_knowledge = f"Regarding the query '{t['query']}', the resolution is: {resolution_text}"
                    if uploaded_image:
                        st.info("Reading image with Gemini Vision...")
                        img = Image.open(uploaded_image)
                        vision_model = genai.GenerativeModel('gemini-1.5-flash')
                        vision_response = vision_model.generate_content([img, "Extract all text and describe the contents of this image."])
                        final_knowledge += f"\n\n--- Additional Context from Image ---\n{vision_response.text}"
                    vector_store.add_documents([Document(page_content=final_knowledge)])
                    vector_store.save_local("faiss_index")
                    for ticket in tickets:
                        if ticket['id'] == t['id']:
                            ticket['status'] = "Closed"
                    save_tickets(tickets)
                    st.success(f"Ticket #{t['id']} closed. The Oracle is now smarter. Please refresh.")
                    st.rerun()

