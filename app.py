import dotenv
dotenv.load_dotenv()

from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFaceHub

llm = HuggingFaceHub(repo_id="declare-lab/flan-alpaca-large", model_kwargs={"temperature":0, "max_length": 512})

template = "what is a good thesis title for this topic: {topic}"

prompt_template = PromptTemplate(
    template = template,
    input_variables=["topic"]
)

from langchain.chains import LLMChain

chain = LLMChain(llm=llm, prompt=prompt_template)

from langchain.callbacks import StreamlitCallbackHandler
import streamlit as st
st.title('Thesis Title Generator')
st.text('Write your topic.')
if prompt := st.chat_input():
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(st.container())
        response = chain.run(prompt, callbacks=[st_callback])
        st.write(response)
