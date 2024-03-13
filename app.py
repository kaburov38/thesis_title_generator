from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFaceHub
from langchain.callbacks import StreamlitCallbackHandler
import streamlit as st
from langchain.chains import LLMChain
import os

os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["api_key"]

llm = HuggingFaceHub(repo_id="google/gemma-7b", model_kwargs={"temperature":0, "max_length": 512})

template = "what is a good thesis title for this topic: {topic}"

prompt_template = PromptTemplate(
    template = template,
    input_variables=["topic"]
)

chain = LLMChain(llm=llm, prompt=prompt_template)

st.title('Thesis Title Generator')
st.text('Write your topic.')
if prompt := st.chat_input():
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(st.container())
        response = chain.run(prompt, callbacks=[st_callback])
        st.write(response)
