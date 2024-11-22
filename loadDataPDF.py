import os
import sys
from dotenv import load_dotenv, find_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# Cargar variables de entorno
_ = load_dotenv(find_dotenv())

# Configuración del entorno
document = os.environ['DOCUMENT']
chunk_size = int(os.environ['CHUNK'])
chunk_overlap = int(os.environ['OVERLAP'])
persist_directory = os.environ['PERSIST_DIRECTORY']

# Leer el documento PDF
loader = PyPDFLoader(document)
pages = loader.load()

# Dividir el documento en fragmentos
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    separators=["\n\n", "\n", "(?<=\. )", " ", "\t", ""]
)
splitted_pages = text_splitter.split_documents(pages)

"""print("Número de segmentos: ", len(splitted_pages))
print("***")      
for i in range(10):
    print("Tamaño del segmento: ",len(splitted_pages[i].page_content))
    #print("Metadata: ", splitted_pages[i].metadata)
    print("Contenido:" , splitted_pages[i].page_content)
    print("-----------------------------------------------")"""

# Crear la base de vectores
embedding = OpenAIEmbeddings()
vectordb = Chroma.from_documents(
    documents=splitted_pages,
    embedding=embedding,
    persist_directory=persist_directory
)

print("Base de datos de vectores creada y persistida en:", persist_directory)
