from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader,PyPDFLoader
from langchain_core.output_parsers import StrOutputParser

llm = HuggingFaceEndpoint(repo_id='meta-llama/Llama-3.1-8B-Instruct',task='text-generation')
model = ChatHuggingFace(llm=llm)

text = PyPDFLoader('text.pdf')
docs = text.load()
doc = "\n\n".join([doc.page_content for doc in docs])
# print(doc)
prompt = PromptTemplate(
    template="What is the teacher name here and what is this assignmwnt about ? \n {text}",
    input_variables=['text']
)
parser=  StrOutputParser()
print(len(doc))
chain = prompt | model | parser
print(chain.invoke({'text':doc}))

