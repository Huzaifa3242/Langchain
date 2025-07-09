from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
load_dotenv()
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser,ResponseSchema
from pydantic import BaseModel,Field
from typing import Optional,Annotated
from langchain_core.output_parsers import StrOutputParser

llm = HuggingFaceEndpoint(repo_id='meta-llama/Llama-3.1-8B-Instruct',task = 'text-generation')
model = ChatHuggingFace(llm = llm)
parser2 = StrOutputParser()
# Schema
scheme = [
    ResponseSchema(name = 'fact 1',description = 'Fact 1 about the topic'),
    ResponseSchema(name = 'fact 2',description = 'Fact 2 about the topic'),
    ResponseSchema(name = 'fact 3',description = 'Fact 3 about the topic'),
    ResponseSchema(name = 'fact 4',description = 'Fact 4 about the topic'),
]
parser = StructuredOutputParser.from_response_schemas(scheme)
template = PromptTemplate(
    template= """Give me the 4 facts about {topic} \n {format_instruction}""",
    input_types={'topic':str},
    partial_variables={'format_instruction':parser.get_format_instructions()}
)
chain = template | model | parser
result = chain.invoke({"topic":"Langchain"})
print(result)
