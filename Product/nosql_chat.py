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
collection=db["products_main"]  #products_main


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
        Below is the collection named "products_main"
        1. 'Shape': Describes the geometric form or outline of the product, influencing its aesthetic appeal and functionality.
        2. 'Range': Refers to the product collection or series it belongs to, indicating related models or versions.
        3. 'UK Brand': The brand originating from the UK that manufactures or distributes the product.
        4. 'Mounting': Indicates the method or type of installation required for the product, such as wall-mounted, floor-mounted, etc.
        5. 'FinishNew': Describes the exterior surface treatment or texture of the product (e.g., polished, matte).
        6. 'Material': This is the product unique id and it is integer
        7. 'Product Title': The official name of the product, used for marketing and identification.
        8. 'Installation': Details the process or requirements for installing the product, including tools and expertise needed.
        9. 'Product category': Defines the broad classification under which the product falls (e.g., kitchen, bathroom, electrical).
        10. 'Pricesegment': Categorizes the product based on its price range (e.g., budget, mid-range, premium).
        11. 'WasteType': Specifies the type of waste management system or compatibility (if applicable).
        12. 'FinishPlus': Highlights any additional finishes or coatings that enhance the product’s durability or appearance.
        13. 'Brand': The name of the manufacturer or distributor that markets the product.
        14. 'Full Description': A detailed description of the product, including its features, benefits, and specifications.
        15. 'Outlet': Refers to the distribution channel or store where the product is sold. ex 1 outlet,  2 outlets
        16. 'Product sub-category': A more specific classification within the broader product category (e.g., under kitchen: faucets, sinks).
        17. 'No_of_Holes': Specifies the number of holes or openings the product requires or has for installation purposes (e.g., sink with two faucet holes).


        This schema provides a information on product attributes. 
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




