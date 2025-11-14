# Website summarization
# (D:\Udemy\Complete_GenAI_Langchain_Huggingface\Python\venv) 
# D:\Udemy\Complete_GenAI_Langchain_Huggingface\Python\35-Youtube video and website summarization>streamlit run 2-app.py

import validators,streamlit as st
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.vectorstores.faiss import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains.retrieval import create_retrieval_chain

## streamlit configuration
st.set_page_config(page_title="LangChain: Summarize Text From Website", page_icon="🦜")
st.title("🦜 LangChain: Summarize Text From Website")
st.subheader('Summarize URL')

## Sidebar for API Key input
st.sidebar.title("Settings")
groq_api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password")
if groq_api_key:
    # llm model
    llm =ChatGroq(model="llama-3.1-8b-instant", groq_api_key=groq_api_key)
else:
    st.warning("⚠️ Please enter a valid GROQ API Key to continue.")
    st.stop()

## URL of any Website or YT video
generic_url=st.text_input("Give URL of any Website, you want the summariation of: ") # https://whc.unesco.org/en/list/252/
if not generic_url:
    st.warning("⚠️ Please enter a valid URL to continue.")
    st.stop()

## Prompt
prompt=ChatPromptTemplate.from_messages(
    [("system", "Please summarize the below speech:"),("user","User question:{input}, Speech: {context}")])

## Summarization
if st.button("Summarize the Content"):
    ## Validate all the inputs
    if not validators.url(generic_url):
        st.error("Please enter a valid Url. It can be any Website url")
    else:
        with st.spinner("Please wait..."):
            docs = WebBaseLoader(web_path=generic_url).load() # https://howrah.gov.in/tourist-place/belur-math-temple/
            final_documents=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=50).split_documents(docs)
            embeddings = HuggingFaceEmbeddings()
            db = FAISS.from_documents(final_documents,embeddings)
            db_retriever = db.as_retriever()
            chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
            rag_chain = create_retrieval_chain(retriever=db_retriever, combine_docs_chain=chain)
            output_summary=rag_chain.invoke({"input":generic_url})
            st.success(output_summary["answer"])
