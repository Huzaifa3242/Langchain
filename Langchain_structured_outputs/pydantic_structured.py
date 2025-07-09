from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from dotenv import load_dotenv
from pydantic import BaseModel,Field
load_dotenv()
llm = HuggingFaceEndpoint(repo_id='meta-llama/Llama-3.1-8B-Instruct',task='text-generation')
model = ChatHuggingFace(llm=llm)

# Pydantic (for validations)
class Review(BaseModel):
    sentiment: str = Field(description = "Return the sentiment of the review as 1 or 0")
    summary: str = Field(description="A brief summary of the review")

structed_model=model.with_structured_output(Review)
try:
    result = structed_model.invoke("The product was good but it is very expensive so you should not buy it")
    print(result)
except Exception as e:
    print(f"Error: {e}")
