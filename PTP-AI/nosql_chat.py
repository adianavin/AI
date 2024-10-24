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


st.title("example")
st.write("ask anything and get answer")
input=st.text_area("enter your question here")

with io.open("PTP-AI/sample.txt","r",encoding="utf-8")as f1:
    sample=f1.read()
    f1.close()

prompt="""
        you are a very intelligent AI assitasnt who is expert in identifying relevant questions from user
        and converting into nosql mongodb agggregation pipeline query.
        Note: You have to just return the query as to use in agggregation pipeline nothing else. Don't return any other thing
        Please use the below schema to write the mongodb queries , dont use any other queries.
        Here's a breakdown of its schema with descriptions for each field:
        Below is the collection named "product_analysis"
        1. **_id**: Unique identifier for the document.
        2. **month_year**: Month and year of the record, in MMM-YYYY format ex JAN-2023, FEB-2024
        3. **Categories_name**: Name of the category.
        4. **Categories_value**: Value representing the category. ex. "Shower combis conc.","Bath Faucets","Showerpipes","Headshowers".
        5. **Distribution Channel_value**: Value representing the distribution channel ex. "Trade", "Global Projects","eCommerce","DIY","After Sales".
        6. **Height_value**: Value representing the product height. ex. "Height-S","Height-M","Height-L".
        7. **Mounting_value**: Value representing the mounting type. ex. "deck mounted","floor-standing","Ceiling","Wall","Flush".
        8. **No of Holes_value**: Value representing the 'Number of Holes' attribute. ex. "1 Hole","4 Holes","3 Holes","2 Holes".
        9. **No of Outlets_value**: Value representing the 'Number of Outlets' attribute. ex. "2 outlets".
        10. **Price Segment_value**: Value representing the price segment.  ex. "Best", "Better", "Good", "Axor".
        11. **Shape_value**: Value representing the product shape. e.g. Round, Softcube, Baton, Square etc.
        12. **Shower Head Size_value**: Value representing the shower head size. ex. "Overheads-Large", "Overheads-Small", "Overheads-Medium".
        13. **Volume**: Volume of the product sold.
        14. **sales**: Total sales in pound for the product.


        This schema provides a attributes wise sales and volume for the given month. 
        Use the below sample_examples to generate your queries perfectly.   

sample_example:

Below are several sample user questions related to the MongoDB document provided, and the corresponding MongoDB aggregation pipeline queries that can be used to fetch the desired data. Please include '$project' stage in each pipeline to have meaningfull output. 
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
Always use Pound symbol for sales.
If the input includes an array, please read all the values and interpret the results clearly. Focus on providing an output that a simple user can easily understand, without delving into technical details. Present the information in a straightforward and approachable manner.
If there is no information available in the output, do not provide any additional information from other resources or from your own knowledge. Simply state that no information is available or no data found with the context.
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
        try:
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
        except Exception as e:
            print(e)
            st.write(response["text"])




