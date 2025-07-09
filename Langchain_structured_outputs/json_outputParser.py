from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
load_dotenv()
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

parser = JsonOutputParser()
llm = HuggingFaceEndpoint(repo_id='meta-llama/Llama-3.1-8B-Instruct',task = 'text-generation')
model = ChatHuggingFace(llm = llm)

template = PromptTemplate(template="""
Give me name age of the person which live in {country}.You can chose any dummy name \n {format_instruction}
""",
input_variables=["country"],
partial_variables={'format_instruction':parser.get_format_instructions()})

chain = template | model | parser
result=chain.invoke({'country':"India"})


print(result)