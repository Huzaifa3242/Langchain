from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableParallel,RunnableSequence
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableBranch,RunnablePassthrough
parser = StrOutputParser()

load_dotenv()
llm = HuggingFaceEndpoint(repo_id='meta-llama/Llama-3.1-8B-Instruct',task = 'text-generation')
model = ChatHuggingFace(llm = llm)
parser = StrOutputParser()
prompt1 = PromptTemplate(
    template = "Generate a report about this {topic}",
    input_variables=['topic']
)
prompt2= PromptTemplate(
    template="Summarize the following text \n {text}",
    input_variables=['text']
)

chain = prompt1 | model | parser
branch= RunnableBranch(
    (lambda x:len(x.split()) > 500,prompt2 | model | parser),
    RunnablePassthrough()
)
chain1 = chain | branch
result = chain1.invoke({'topic':"Langchain"})
print(result)