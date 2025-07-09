from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from dotenv import load_dotenv
from typing import TypedDict,Annotated,Optional,Literal
load_dotenv()
llm = HuggingFaceEndpoint(repo_id='meta-llama/Llama-3.1-8B-Instruct',task='text-generation')
model = ChatHuggingFace(llm=llm)

# Typeddict (for represntational purposes not validations)
class Review(TypedDict):
    sentiment:Annotated[Literal[0,1],"Return  sentiment of the review"]
    summary:Annotated[str,"A brief summary of the review"]
    # key_themes:Annotated[list[str],"Here retun the list"]
    # pros:Annotated[Optional[list[str]];""]

structed_model=model.with_structured_output(Review)

result=structed_model.invoke("the product was good but it is very expensive so you should not buy it")
print(result)