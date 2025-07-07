from langchain_core.prompts import ChatPromptTemplate
message_temp = ChatPromptTemplate(
    [('system',"You are {domain} expert"),
    ('human', "Tell me about {topic} in detail")]
)
prompt = message_temp.invoke({
   "domain":'cricket',
   'topic':"dusra"
})
print(prompt)