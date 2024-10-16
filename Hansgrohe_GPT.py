GROQ_API_KEY="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"
import pandas as pd
from langchain_groq import ChatGroq
import streamlit as st
from pandasai import SmartDataframe
from langchain_community.llms import Ollama

def chat_with_csv(df, query):
    # llm = Ollama(
    # model="llama3",
    # temperature=0  # Increased temperature to encourage more diverse responses
    # )

    # llm = Ollama(
    # model="gemma2:2B",
    # temperature=0  # Increased temperature to encourage more diverse responses
    # )
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY, model_name="llama3-8b-8192", temperature=0
    )
    #print(df)
    pandas_ai = SmartDataframe(df,config={"llm":llm})
    print(pandas_ai)
    result = pandas_ai.chat(query)
    return result

st.set_page_config(layout="wide")
st.title("Welcome to LLM powered Chat")
input_csvs = st.sidebar.file_uploader("Upload your CSV files", type=["csv"], accept_multiple_files=True)

if(input_csvs):
    selected_file = st.selectbox("Select a CSV file", [file.name for file in input_csvs])
    selected_index = [file.name for file in input_csvs].index(selected_file)
    st.info("CSV uploaded successfully")
    data = pd.read_csv(input_csvs[selected_index])
    st.dataframe(data.head(3), use_container_width=True)

    st.info("Chat below")
    input_text = st.text_area("Enter the query")
    if(input_text):
        if st.button("Chat with CSV"):
            st.info("Your query:" + input_text)
            result = chat_with_csv(data, input_text)
            st.success(result)



