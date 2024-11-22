import os
import openai
from dotenv import load_dotenv, find_dotenv
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import LLMChain

import streamlit as st
from streamlit_chat import message
from PIL import Image


def create_memory():
    return ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        input_key='question',
        output_key='answer'
    )

_ = load_dotenv(find_dotenv())
openai.api_key = os.environ['OPENAI_API_KEY']
faq = []

# Configuración de la interfaz
st.set_page_config(page_title="Chatbot con CSV")
st.markdown("<h3 style='text-align:center; color: #009AD9'>GPT Infinitum</h3>", unsafe_allow_html=True)
##container del chat history
chatContainer = st.container()
##Container TextBox
inputContainer = st.container()
botAvatarPath = "./src/Cablin.png"
#botAvatarUser = "./src/worker.jpg"
botAvatarUser = Image.open("./src/worker.jpg") 
botAvatarUser = botAvatarUser.resize((70, 70))



with inputContainer:
    question = st.chat_input(placeholder="¿En qué puedo ayudarte?", key="inputQuestion")
    st.markdown('<div class="messages-container">', unsafe_allow_html=True)
    st.markdown("<h5 class='message-init' color: #11111'>💠Sugerido</h3>", unsafe_allow_html=True)
    st.markdown("<p class='message-init' color: #11111'>🔹¿Qué es Telmex y qué ofrece?</p>", unsafe_allow_html=True)
    st.markdown("<p class='message-init' color: #11111'>🔹Desempeño de Módems y ONTs (Ranking y Tasa de sustitución)</p>", unsafe_allow_html=True)
    st.markdown("<p class='message-init' color: #11111'>🔹Información Operativa (Tecnologias, COPES, OS, etc.)</p>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    #with st.form(key="formQuestion",clear_on_submit=True):
        #col1, col2 = st.columns([3, 1])  # Ajustar el tamaño de las columnas a tus necesidades
        #with col1:
            #question = st.text_input("Tu pregunta:", key="inputQuestion")
        #with col2:
            #st.markdown('<div class="button-margin"></div>', unsafe_allow_html=True)
            #submitButton = st.form_submit_button(label="Enviar")



# Cargar la base de vectores
persist_directory = os.environ['PERSIST_DIRECTORY']
embedding = OpenAIEmbeddings()
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)

# Crear plantilla de prompt
template = """"

Actúa como un asistente virtual. Utiliza la siguiente información de contexto para responder a la pregunta al final.

Importante:
- Todas las respuestas deberan estar en idioma Español
- Usa un tono amable y profesional en tus respuestas.
- Si no conoces la respuesta o no está en la información de contexto, utiliza frases como: 'No tengo la información que buscas', 'Lamento no poder ayudarte, pero no tengo entrenamiento para responder esto' o 'No tengo la respuesta, pero ¿puedo ayudarte con algo más?'.
- Utiliza las preguntas y respuestas anteriores como contexto adicional para ofrecer respuestas más relevantes.
- Después de cada respuesta proporcionada, sugiere 2 preguntas relacionadas que el usuario podría hacer en función de la información entregada.El formato de estas dos preguntas que se en forma de lista con un subtitulo que diga "Preguntas Sugeridas" Evita esto si el usuario se despide
- Despídete de manera amable y formal, usando frases como: 'Fue un placer ayudarte', 'Hasta luego, espero haberte ayudado', o 'Un placer asistirte; si necesitas algo más, no dudes en buscarme'.
- El contexto lo encontraras entre comillas simples

'''
{context}
'''

PREGUNTA: {question}
RESPUESTA DE AYUDA:"""

QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
retriever = vectordb.as_retriever(search_kwargs={"k": 10})
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
llm_chain = LLMChain(llm=llm, prompt=QA_CHAIN_PROMPT)



memory = create_memory()

qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    return_source_documents=True,
    chain_type="stuff",
    combine_docs_chain_kwargs={'prompt': QA_CHAIN_PROMPT}
)

if 'responses' not in st.session_state:
    st.session_state['responses'] = []

if 'requests' not in st.session_state:
    st.session_state['requests'] = []


if question:
    with st.spinner("Escribiendo..."):
        if question == "Olvida conversación":
            memory = create_memory()
            qa.memory = memory
            st.session_state.requests.append(question)
            st.session_state.responses.append("He olvidado la conversación. ¿En qué puedo ayudarte ahora?")
        else:
            result = qa.invoke({"question": question})
            for doc in result['source_documents']:
                print(doc)
                print("-----------------------------------------------")
            print("******************************************************")
            st.session_state.requests.append(question)
            response =  result['answer'].split("Preguntas Sugeridas:\n")
            
            
            faq = response[1]
            response = response[0]
            faq = faq.split("\n")
            st.markdown("""
                <style>
                .full-width-button .stButton button {
                    width: 100%;
                }
                </style>
                """, unsafe_allow_html=True)
            if(len(faq) > 0):
                st.markdown(f"<p class='message' color: #11111'>🔹{faq[0]}</p>", unsafe_allow_html=True)
                st.markdown(f"<p class='message' color: #11111'>🔹{faq[1]}</p>", unsafe_allow_html=True)
            st.session_state.responses.append(response)

with chatContainer:
    if st.session_state['responses']:
        for i in range(len(st.session_state['responses'])):
            if i < len(st.session_state['requests']):
                st.chat_message("user", avatar=botAvatarUser).write(st.session_state['requests'][i])
                #message(
                    #st.session_state['requests'][i],
                    #is_user=True,
                    #key=str(i)+"_user")
            #message(
                #st.session_state['responses'][i],
                #key=str(i),
                #avatar_style=botAvatarPath)
                st.chat_message("assistant", avatar=botAvatarPath).write(st.session_state['responses'][i])
                st.markdown("""
                <style>
                .message-init {
                    display: none;
                }
                .messages-container{
                    display: none;      
                }
                .message{
                    position: relative;
                    inset-block: -5rem;
                }
                </style>
                """, unsafe_allow_html=True)
                