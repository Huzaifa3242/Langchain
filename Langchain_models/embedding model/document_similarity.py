from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

doc = [
    "Virat Kohli is known for his aggressive batting and unmatched consistency in all formats.",
    "Babar Azam is admired for his elegant stroke play and calm leadership on the field.",
    "Shaheen Afridi is a fiery left-arm pacer famous for his deadly yorkers and early breakthroughs."
]

query = "Tell me about Babar Azam"

doc_embeddings = embed_model.embed_documents(doc)
query_embeddings = embed_model.embed_query(query)

scores = cosine_similarity([query_embeddings], doc_embeddings)[0]
index,score = sorted(list(enumerate(scores)),key=lambda x:x[1])[-1]

print(query)
print(doc[index])