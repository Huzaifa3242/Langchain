from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage,SystemMessage,AIMessage

load_dotenv()
message = [
    SystemMessage(content='You are a helpful AI Assitant')
]
llm = HuggingFaceEndpoint(repo_id='meta-llama/Llama-3.1-8B-Instruct',task='text-generation')
model = ChatHuggingFace(llm=llm)
while(True):
    x= input("You :")
    message.append(HumanMessage(content=x))
    if x == "exit":
        break
    result = model.invoke(message)
    message.append(AIMessage(content=result.content))
    print("AI :",result.content)
    print(message)