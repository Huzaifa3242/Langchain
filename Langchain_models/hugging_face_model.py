from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from  dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(repo_id='meta-llama/Llama-3.1-8B-Instruct',task='text-generation')

model = ChatHuggingFace(llm=llm)

result=model.invoke('What is the capital of Pakistan?')

print(result.content)