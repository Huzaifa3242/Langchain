from langchain.text_splitter import RecursiveCharacterTextSplitter,Language

text = """
from langchain_huggingface import ChatHuggingFace ,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader,PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
loader = DirectoryLoader(
    path = 'Dummy pdfs',
    glob = '*.pdf',
    loader_cls=PyPDFLoader
)
docs = loader.load()
splitter = CharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 0,
    separator=''
)
splitter_docs=splitter.split_documents(docs)
load_dotenv()
"""
splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size = 100,
    chunk_overlap =0
)
slitter_txt = splitter.split_text(text)
print(slitter_txt)