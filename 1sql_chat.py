GROQ_API_KEY="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"

from langchain_community.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import HumanMessagePromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

llm = ChatGroq(
        groq_api_key=GROQ_API_KEY, model_name="llama3-8b-8192", temperature=0
)

host = 'localhost'
port = '3306'
username = 'lantern'
password = 'lantern'
database_schema = 'lantern'
mysql_uri = f"mysql+pymysql://{username}:{password}@{host}:{port}/{database_schema}"

db = SQLDatabase.from_uri(mysql_uri, include_tables=['jobs'],sample_rows_in_table_info=2)

print("Database connected")
db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)

#db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)

def retrieve_from_db(query: str) -> str:
    db_context = db_chain.invoke(query)
    db_context = db_context['result'].strip()
    return db_context

def generate(query: str) -> str:
    db_context = retrieve_from_db(query)
    
    system_message = """You are a professional representative of an employment agency.
        You have to answer user's queries and provide relevant information to help in their job search. 
        Example:
        
        Input:
        Where are the most number of jobs for an English Teacher in Canada?
        
        Context:
        The most number of jobs for an English Teacher in Canada is in the following cities:
        1. Ontario
        2. British Columbia
        
        Output:
        The most number of jobs for an English Teacher in Canada is in Toronto and British Columbia
        """
    
    human_qry_template = HumanMessagePromptTemplate.from_template(
        """Input:
        {human_input}
        
        Context:
        {db_context}
        
        Output:
        """
    )
    messages = [
      SystemMessage(content=system_message),
      human_qry_template.format(human_input=query, db_context=db_context)
    ]
    response = llm.invoke(messages).content
    return response

answer = generate("what are the jobs available in KPMG?")
print(answer)