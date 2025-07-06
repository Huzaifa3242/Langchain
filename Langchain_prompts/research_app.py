import streamlit as st 
from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate,load_prompt
from dotenv import load_dotenv
load_dotenv()

st.header("Simple Research App")

# Select paper input
paper_input = st.selectbox(
    "Select the Research Paper:",[
        "Attention is all you need",
        "Quantum Computing Basics",
        "AI in Healthcare",
        "Cryptography Techniques",
        "Neural Networks and Deep Learning"
    ]
)

# Select explanation style
style_input = st.selectbox(
    "Choose Explanation Style:",[
        "Technical and Formal",
        "Beginner-Friendly",
        "Conversational",
        "Bullet Point Summary",
        "Academic Tone"
    ]
)

# Select explanation length
length_input = st.selectbox(
    "Choose Summary Length:",
    [
        "Very Short (1-2 sentences)",
        "Short (1 paragraph)",
        "Medium (2-3 paragraphs)",
        "Long (Full page)",
        "Detailed (With examples)"
    ]
)
llm = HuggingFaceEndpoint(repo_id='meta-llama/Llama-3.1-8B-Instruct',task='text-generation')
model = ChatHuggingFace(llm=llm)

template = load_prompt('template.json')

# user_input = st.text_input("Enter your prompt here")
if st.button("Summarize"):
    # Using chains (Advance way)
    chain = template | model
    result = chain.invoke({
    "paper_input":paper_input,
    'style_input':style_input,
    'length_input':length_input}
    )
    st.write(result.content)
    # Simple way
    #  prompt = template.invoke({
    # "paper_input":paper_input,
    # 'style_input':style_input,
    # 'length_input':length_input})
    #  result = model.invoke(prompt)
   



