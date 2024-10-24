GROQ_API_KEY="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"

import streamlit as st
from pymongo import MongoClient
import urllib,io,json
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

#llm=ChatOpenAI(model="gpt-4",temperature=0.0)
#mongo client
llm = ChatGroq(
        groq_api_key=GROQ_API_KEY, model_name="llama3-8b-8192", temperature=0
)



client=MongoClient("mongodb://localhost:27017/")
db=client["hansgrohe"]
collection=db["product_analysis"]  #products_main


st.title("LLM Powered Chat")
st.write("ask anything and get answer")
input=st.text_area("enter your question here")

with io.open("sample.txt","r",encoding="utf-8")as f1:
    sample=f1.read()
    f1.close()

prompt="""
        you are a very intelligent AI assitasnt who is expert in identifying relevant questions from user
        and converting into nosql mongodb agggregation pipeline query.
        Note: You have to just return the query as to use in agggregation pipeline nothing else. Don't return any other thing
        Please use the below schema to write the mongodb queries , dont use any other queries.
       schema:
       the mentioned mogbodb collection talks about various products (material) offered by Hansgrohe and AXOR. 
       The schema for this document represents the structure of the data, 
       describing various properties related to the product (material), 
       product sub-category, Pricesegment, Shape, and additional features. 
       your job is to get python code for the user question. 
    

       Here's a breakdown of its schema with descriptions for each field:
       Below is the collection named "product_analysis"

1. **_id**: Unique identifier for the document
2. **core_filter_text**: Concatenated string using "~" representing key product attributes 
3. **core_filter_value**: Concatenated string using "~" representing values for each product attribute
4. **core_filter_field**: Fields that are part of the filtering criteria
5. **sales**: Total sales for the product in pound
6. **Volume**: Quantity of product sold in actual
7. **month_year**: Month and year of the transaction ex JAN-2023, FEB-2024 etc.
8. **material_ids**: Comma-separated list of material IDs related to the product
9. **Distribution Channel_name**: name for the distribution channel e.g. "Trade"
10. **Distribution Channel_value**: Value representing the distribution channel
12. **Categories_value**: Value representing the product category
13. **Price Segment_name**: name for the price segment ex. "Best", "Better", "Good", "Axor"
15. **No of Holes_name**: product attribute number of holes
16. **No of Holes_value**: product attribute number of holes 
17. **Shape_name**: product attribute e.g. Round, Softcube, Baton, Square etc


This schema provides a comprehensive view of the data structure for Hansgrohe Products (Material) in MongoDB. 
Please note that MongoDB  pipeline always run aggregation on "products_main" collection hence write the pipeline in that way.
use the below sample_examples to generate your queries perfectly.   

sample_example:

Below are several sample user questions related to the MongoDB document provided, 
and the corresponding MongoDB aggregation pipeline queries that can be used to fetch the desired data.
Use them wisely.

sample_question: {sample}
As an expert you must use them whenever required.
Note: You have to just return the query nothing else. Don't return any additional text with the query.
Please follow this strictly
input:{question}
output:
"""

result_prompt = """
You have to generate answer based on the question and output.
input:{question} {output}
"""

query_with_prompt=PromptTemplate(
    template=prompt,
    input_variables=["question","sample"]
)
result_formatting_prompt=PromptTemplate(
    template=result_prompt,
    input_variables=["question","output"]
)
llmchain=LLMChain(llm=llm,prompt=query_with_prompt,verbose=True)
result_llmchain = LLMChain(llm=llm, prompt=result_formatting_prompt, verbose=True)

if input is not None:
    button=st.button("Submit")
    if button:
        response=llmchain.invoke({
            "question":input,
            "sample":sample
        })
        print(response["text"])
        query=json.loads(response["text"])
        results=collection.aggregate(query)
        result_list = list(results)   
        for result in results:
            st.write(result)
        response_result=result_llmchain.invoke({
            "question":input,
            "output":result_list
        })
        st.write(response_result["text"])



