import os
import pandas as pd
from dotenv import load_dotenv, find_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document

# Cargar variables de entorno
_ = load_dotenv(find_dotenv())

# Configuración del entorno
csv_file = os.environ['DOCUMENT']
chunk_size = int(os.environ['CHUNK'])
chunk_overlap = int(os.environ['OVERLAP'])
persist_directory = os.environ['PERSIST_DIRECTORY']

# Leer el archivo CSV
df = pd.read_csv(csv_file,encoding="latin-1")


# Convertir cada fila en un documento con las columnas como contenido estructurado
documents = []
content = "El catalgo de Copes contiene información de la compañia Telmex. El cual contiene información como las siglas o clave de cope, el nombre del cope, en qué área se encuentra y quien es su responsable. El catalogo de copes consta de 250 copes repartidos en diversas áreas de la republica Mexicana. A continuación se presenta el catalogo de copes:\n "
metadata = {
    "DOCUMENT": "CATALOGO DE COPES"
}
documents.append(Document(page_content=content, metadata=metadata))
for index, row in df.iterrows():
    content = f"{index+1}. El cope '{row['SIGLAS_DE_COPE']}' tiene el nombre de {row['NOMBRE_COPE']}, se encuentra en {row['AREA_COPE']} y su responsable es {row['RESPONSABLE_COPE']}"
    metadata = {
        "DOCUMENT": "CATALOGO DE COPES",
        "COPE": row['SIGLAS_DE_COPE']
    }
        
    documents.append(Document(page_content=content, metadata=metadata))




# Dividir cada documento en fragmentos si son muy largos
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    separators=["\n\n", "\n", "(?<=\. )", " ", "\t", ""]
)
splitted_documents = text_splitter.split_documents(documents)



# Crear la base de vectores
embedding = OpenAIEmbeddings()
vectordb = Chroma.from_documents(
    documents=splitted_documents,
    embedding=embedding,
    persist_directory=persist_directory
)

print("Base de datos de vectores creada y persistida en:", persist_directory)
