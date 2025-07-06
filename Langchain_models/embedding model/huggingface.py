from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
load_dotenv()
embedings = HuggingFaceEmbeddings(model_name ="sentence-transformers/all-MiniLM-L6-v2")
text = "Islamabad is capital of Pakistan"
vector = embedings.embed_query(text)
print(str(vector))