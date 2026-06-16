import streamlit as st
import pandas as pd
import plotly.express as px
import os
import json
from PIL import Image
import google.generativeai as genai

# Corrected LangChain Imports for the latest versions
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

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
        # Initial ingestion of dummy text
        with open("dummy_data/HR_Policy.txt", "r") as f:
            text = f.read()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = [Document(page_content=chunk) for chunk in splitter.split_text(text)]
        vector_store = FAISS.from_documents(docs, embeddings)
        vector_store.save_local("faiss_index")
        return vector_store

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
st.markdown("*Institutional Intelligence & Knowledge Synthesis Engine*")

tab1, tab2, tab3 = st.tabs(["💬 Employee Chat", "📊 Finance Visualizer", "👔 CXO Bridge"])

# --- TAB 1: EMPLOYEE CHAT ---
with tab1:
    st.subheader("Ask the Oracle")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input("Ask about policies, structures, or say 'escalate'"):
        st.chat_message("user").write(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Retrieval Chain Setup
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        system_prompt = (
            "You are the Vivriti Oracle. Use the provided context to answer the user's question. "
            "If the answer is not in the context, strictly reply with: 'CONFIDENCE_LOW'."
            "\n\nContext:\n{context}"
        )
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        qa_chain = create_stuff_documents_chain(llm, prompt_template)
        rag_chain = create_retrieval_chain(retriever, qa_chain)

        response = rag_chain.invoke({"input": prompt})
        answer = response["answer"]

        # Handle Escalation Workflow
        if "CONFIDENCE_LOW" in answer:
            st.chat_message("assistant").write("I'm sorry, I don't have enough information to answer this with high confidence. Please route this to an expert.")

            with st.form("escalation_form"):
                st.write("### 🚨 Route to Expert")
                dept = st.selectbox("Select Department", ["HR", "Credit", "Capital Markets", "Tech"])
                query = st.text_area("Your Query", value=prompt)
                submitted = st.form_submit_button("Raise CXO Ticket & Show Contact")

                if submitted:
                    tickets = load_tickets()
                    tickets.append({"id": len(tickets)+1, "dept": dept, "query": query, "status": "Open"})
                    save_tickets(tickets)

                    st.success("Ticket raised!")
                    st.info(f"**Contact Info:**\n* Department Head: {dept}_head@vivriti.com\n* Teams: ping {dept} Lead")
        else:
            st.chat_message("assistant").write(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})


# --- TAB 2: FINANCE VISUALIZER ---
with tab2:
    st.subheader("Interactive Cash Flow Visualizer")
    st.markdown("Upload a Cash Flow model or view the dummy project data.")

    df = pd.read_csv("dummy_data/Project_CashFlows.csv")
    df['Net_Cash_Flow'] = df['Cash_In'] + df['Cash_Out']
    df['Cumulative_Cash'] = df['Net_Cash_Flow'].cumsum()

    col1, col2 = st.columns([2, 1])
    with col1:
        # "Finance Professor" Plotly Chart
        fig = px.bar(df, x='Month', y=['Cash_In', 'Cash_Out'],
                     title="Monthly Cash Inflows vs Outflows",
                     barmode='group',
                     color_discrete_sequence=['#2ecc71', '#e74c3c'])
        st.plotly_chart(fig, use_container_width=True)

        fig2 = px.line(df, x='Month', y='Cumulative_Cash',
                       title="Cumulative Cash Flow Curve (Breakeven Analysis)",
                       markers=True)
        fig2.add_hline(y=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        st.markdown("### The Bottom Line 💡")
        breakeven_month = df[df['Cumulative_Cash'] > 0]['Month'].min()
        if pd.notna(breakeven_month):
            st.success(f"**Breakeven achieved in Month {breakeven_month}.**")
        else:
            st.error("**Project does not break even in the projected timeline.**")

        st.markdown("### The Tape (Raw Data)")
        st.dataframe(df, hide_index=True)


# --- TAB 3: CXO BRIDGE ---
with tab3:
    st.subheader("CXO Dashboard: Resolve & Teach")
    tickets = load_tickets()
    open_tickets = [t for t in tickets if t["status"] == "Open"]

    if not open_tickets:
        st.success("No open escalations. The Oracle is up to date!")
    else:
        for t in open_tickets:
            with st.expander(f"Ticket #{t['id']} | Dept: {t['dept']} | Query: {t['query']}"):
                resolution_text = st.text_area("Resolution Notes (Type the answer here)", key=f"res_{t['id']}")
                uploaded_image = st.file_uploader("Upload supporting screenshot (optional)", type=["png", "jpg", "jpeg"], key=f"img_{t['id']}")

                if st.button("Close Ticket & Update Oracle Knowledge Base", key=f"btn_{t['id']}"):
                    extracted_text = ""

                    # 1. Image OCR using Gemini Native Vision
                    if uploaded_image:
                        st.info("Extracting text from image via Gemini Vision...")
                        img = Image.open(uploaded_image)
                        vision_model = genai.GenerativeModel('gemini-1.5-flash')
                        vision_response = vision_model.generate_content([img, "Extract all text and describe any process/flowchart in this image comprehensively."])
                        extracted_text = vision_response.text

                    # 2. Combine and add to FAISS
                    final_knowledge = f"Query: {t['query']}\nResolution: {resolution_text}\nImage Data: {extracted_text}"

                    new_doc = Document(page_content=final_knowledge)
                    vector_store.add_documents([new_doc])
                    vector_store.save_local("faiss_index")

                    # 3. Close ticket
                    for idx, ticket in enumerate(tickets):
                        if ticket['id'] == t['id']:
                            tickets[idx]['status'] = "Closed"
                    save_tickets(tickets)

                    st.success("Knowledge ingested! The Oracle now knows the answer to this. Refresh the page to see changes.")
                    st.rerun()

