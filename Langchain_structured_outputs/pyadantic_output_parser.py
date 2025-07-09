from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
load_dotenv()
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel,Field
from langchain_core.output_parsers import PydanticOutputParser

class Parserr(BaseModel):
    fact_1:str = Field(description="Fact 1 about the topic")
    fact_2:str = Field(description="Fact 2 about the topic")
    fact_3:str = Field(description="Fact 3 about the topic")
    fact_4:str = Field(description="Fact 4 about the topic")

parser = PydanticOutputParser(pydantic_object=Parserr)
              
llm = HuggingFaceEndpoint(repo_id='meta-llama/Llama-3.1-8B-Instruct',task = 'text-generation')
model = ChatHuggingFace(llm = llm)

template = PromptTemplate(
    template= """Give me the 4 facts about {topic} \n {format_instruction}""",
    input_types={'topic':str},
    partial_variables={'format_instruction':parser.get_format_instructions()}
)
chain = template | model | parser
result = chain.invoke({"topic":"Langchain"})
print(result)
