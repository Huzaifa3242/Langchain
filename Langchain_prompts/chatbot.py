from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()
message = []
llm = HuggingFaceEndpoint(repo_id='meta-llama/Llama-3.1-8B-Instruct',task='text-generation')
model = ChatHuggingFace(llm=llm)
while(True):
    x= input("You :")
    message.append(x)
    if x == "exit":
        break
    print("Message before ai response:",message)
    result = model.invoke(message)
    message.append(result.content)
    print("Message after ai response:",message)
    print("AI :",result.content)
    