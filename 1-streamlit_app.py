# Website summarization
# (D:\Udemy\Complete_GenAI_Langchain_Huggingface\Python\venv) 
# D:\Udemy\Complete_GenAI_Langchain_Huggingface\Python\35-Youtube video and website summarization>streamlit run 2-app.py

import validators,streamlit as st #type:ignore
from langchain.prompts import PromptTemplate #type:ignore
from langchain_groq import ChatGroq #type:ignore
from langchain.chains.summarize import load_summarize_chain #type:ignore
from langchain_community.document_loaders import UnstructuredURLLoader #type:ignore
from langchain.text_splitter import RecursiveCharacterTextSplitter #type:ignore
import traceback #type:ignore

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
chunks_prompt="""
Please summarize the below speech:
Speech:`{text}'
Summary:
"""
map_prompt_template=PromptTemplate(input_variables=['text'], template=chunks_prompt)
final_prompt='''
Provide the final summary of the entire article with these important points.
Speech:{text}
'''
final_prompt_template=PromptTemplate(input_variables=['text'],template=final_prompt)

## Summarization
if st.button("Summarize the Content"):
    ## Validate all the inputs
    if not validators.url(generic_url):
        st.error("Please enter a valid Url. It can be any Website url")
    else:
        try:
            with st.spinner("Please wait..."):
                ## loading the website data
                loader=UnstructuredURLLoader(urls=[generic_url],ssl_verify=False,
                                                 headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"}) # from which browser, u r trying to fetch the url.
                docs=loader.load()
                final_documents=RecursiveCharacterTextSplitter(chunk_size=2000,chunk_overlap=100).split_documents(docs)
                ## Chain For Summarization
                chain=load_summarize_chain(llm=llm,chain_type="map_reduce",map_prompt=map_prompt_template,combine_prompt=final_prompt_template,verbose=True)
                output_summary=chain.run(final_documents)
                st.success(output_summary)
        except Exception as e:
            st.exception(e)
            traceback.print_exc()
