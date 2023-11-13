__version__ = "0.0.1"
app_name = "GitChatGPT"

# BOILERPLATE

import streamlit as st

st.set_page_config(
    layout='centered',
    page_title=f'{app_name} {__version__}'
)

ss = st.session_state
if 'debug' not in ss: ss['debug'] = {}

#st.write(f'<style>{css.v1}</style>', unsafe_allow_html=True)
header1 = st.empty()  # for errors / messages
header2 = st.empty()  # for errors / messages
header3 = st.empty()  # for errors / messages

# IMPORTS

# General modules
import os
import time

# Modules
from dotenv import load_dotenv
from langchain.chains import (
    RetrievalQA,
    ConversationalRetrievalChain
)

from langchain.embeddings import (
    HuggingFaceEmbeddings,
    OpenAIEmbeddings
)

from langchain.prompts import PromptTemplate
from langchain.vectorstores import (
    Chroma,
    DeepLake
)

from common.databases.chroma import get as ChromaAdapter
from common.databases.deeplake import get as DeepLakeAdapter

# Utils
from common import config
from common.llm import (
    load_llm,
    set_llm
)

from common.git import (
    get_repository_url,
    get_repository,
    get_repository_name,
    remove_repository
)

# Rich
from rich.console import Console
console = Console()

chat_history = []

if not load_dotenv():
    console.print(f"Could not load the .env file in the root of the project, or it is empty. Please check if it exists and is readable.", style="red")
    exit(1)

def step_1(selected_model):
    console.print(f"Selected model: {selected_model}", style="yellow")

    config.check_directories()

    # Switch the env file.
    config.switch_dotenv(selected_model)

    # Skip the model directory check for OpenAI and Huggingface
    model = config.get('MODEL')

    llm_name = str(config.get('MODEL_TYPE')).lower()

    # Set LLM variables.
    (llm_name, model_path, local_model) = set_llm(model, config.get('MODEL_TYPE'), config.get('MODEL_PATH'))

    st.session_state.local_model = local_model
    st.session_state.llm_name = llm_name
    st.session_state.model = model
    st.session_state.model_path = model_path

    return selected_model

def step_2(selected_database):
    console.print(f"Selected database: {selected_database}", style="yellow")

    # Override the vector db by the one from the user selection.
    os.environ['VECTOR_DB'] = selected_database

    # Set the default Vector DB from config.
    os.environ['VECTOR_DB'] = config.get('DEFAULT_VECTOR_DB')
    console.print(f"{config.get('VECTOR_DB')} has been set as your Vector Database.", style="green")
    print("")

    # The base path of the Vector Database.
    st.session_state.db_base_path = os.path.join(config.get('DB_PATH'), str(config.get('VECTOR_DB')).lower())

    return selected_database

def step_3(selected_embed_source):
    console.print(f"Selected embedding source: {selected_embed_source}", style="yellow")
    st.session_state.selected_embed_source = selected_embed_source

    return selected_embed_source

def step_4(repository_url):
    st.session_state.override_action = "continue with the current database"

    if repository_url.lower() not in ['', ' ']:
        repository_url = get_repository_url(st.session_state['repository_url'])
        db_name = get_repository_name(repository_url)
        db_path = os.path.join(st.session_state['db_base_path'], db_name)

        # If the vector database already exists we ask the user if he would like to continue or override the database.
        if os.path.exists(db_path):
            st.subheader("REPOSITORY ALREADY EXISTS:")
            override_action = st.selectbox(
                label=" Override Action",
                options=[
                    "Please select an option",
                    "Continue with the current database",
                    "Override the current database"
                ]
            )
            print("")

            documents_path = get_repository(
                repository_url=repository_url,
                clone=False
            )

            # When the user choose for override then remove the repository and vector database.
            if override_action.lower() == "override the current database":
                try:
                    remove_repository(db_path)
                except AssertionError as error:
                    console.print("An error occurred:", type(error).__name__, "–", error, style="red")
                    console.print(
                        "Make sure you have closed all terminals running GitChatGPT and have closed all connections to the database.",
                        style="red")
                    exit(1)

                try:
                    remove_repository(documents_path)
                except AssertionError as error:
                    console.print("An error occurred:", type(error).__name__, "–", error, style="red")
                    console.print(
                        "Make sure you have closed all terminals running GitChatGPT and have closed all connections to the database.",
                        style="red")
                    exit(1)

                print("")
                console.print("Cloning the repository 🚀💻", style="yellow")
                documents_path = get_repository(
                    repository_url=repository_url,
                    clone=True
                )

            st.session_state.override_action = override_action
        else:
            console.print("Cloning the repository 🚀💻", style="yellow")
            documents_path = get_repository(
                repository_url=repository_url,
                clone=True
            )
    else:
        documents_path = 'documents'
        db_name = documents_path
        db_path = os.path.join(st.session_state['db_base_path'], db_name)

    if st.session_state['llm_name'] == 'openai' and config.get('OPENAI_EMBEDDINGS'):
        embeddings = OpenAIEmbeddings(
            api_key=config.get('OPENAI_API_KEY')
        )
    else:
        # Create embeddings.
        embeddings_kwargs = {'device': 'cuda'} if config.get('GPU_ENABLED') and st.session_state['llm_name'] != 'gpt4all' else {}
        embeddings = HuggingFaceEmbeddings(
            model_name=config.get('EMBEDDINGS_MODEL_NAME'),
            model_kwargs=embeddings_kwargs
        )

    with st.spinner('Please wait a bit...'):
        match str(config.get('VECTOR_DB')).lower():
            case "chroma":
                db: Chroma = ChromaAdapter(
                    st.session_state['override_action'],
                    documents_path,
                    db_path,
                    db_name,
                    embeddings
                )

            case "deeplake":
                db: DeepLake = DeepLakeAdapter(
                    st.session_state['override_action'],
                    documents_path,
                    db_path,
                    db_name,
                    embeddings
                )

            case _default:
                # Raise an exception if the model_type is not supported
                console.print(f"Vector DB {config.get('VECTOR_STORE')} is not supported.", style="red")
                exit(1)

        # Update the session state with the entered name
        st.session_state.db = db

def main():
    # LAYOUT
    with st.sidebar:
        llm_providers = config.get_list('LLM_PROVIDERS')
        llm_providers.insert(0, "Please select an option")
        selected_model = st.selectbox(
            label=" LLM Provider",
            options=llm_providers
        )

        # STEP (2)
        vector_databases = config.get_list('VECTOR_DB_PROVIDERS')
        if len(vector_databases) > 1:
            vector_databases.insert(0, "Please select an option")
            selected_database = st.selectbox(
                label=" Vector Database",
                options=vector_databases
            )
        else:
            selected_database = 0

        selected_embed_source = st.selectbox(
            label=" Embedding Type",
            options=["Please select an option", "Github Repository", "Documents"]
        )

        if selected_embed_source == "Github Repository":
            repository_url = st.text_input("Enter the Github repository url")
            st.session_state.repository_url = repository_url

        save_settings = st.button(label="Start Embedding")
        if save_settings:
            step_1(selected_model)
            step_2(selected_database)
            step_3(selected_embed_source)
            step_4(repository_url)

    chat()

def chat():
    question = st.chat_input(
        'What would you like to know about this repository?'
    )

    if 'llm_name' in st.session_state and 'model' in st.session_state and 'model_path' in st.session_state and 'db' in st.session_state:
        with st.spinner("Hold tight for a moment – I might not be super speedy (yet), but I'm generating a response for you! ⏳✨"):
            # GET LLM
            llm = load_llm(
                llm_name=st.session_state['llm_name'],
                model=st.session_state['model'],
                model_path=st.session_state['model_path'],
                stream=False
            )

            # Set the data as retriever
            retriever = st.session_state['db'].as_retriever(
                search_kwargs={"k": int(config.get('TARGET_SOURCE_CHUNKS'))}
            )

            # Retrieval QA
            prompt = PromptTemplate(
                template=config.prompt_template,
                input_variables=["context", "question"]
            )

            qa_type = "Conversational"
            if qa_type == "Conversational":
                qa = ConversationalRetrievalChain.from_llm(
                    llm=llm,
                    retriever=retriever,
                    condense_question_prompt=prompt
                )
            else:
                chain_type_kwargs = {"prompt": prompt} if config.get('PROMPT_ENABLED') else {}
                qa = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=retriever,
                    chain_type_kwargs=chain_type_kwargs,
                    return_source_documents=False
                )

            if question:
                # Ask for user input
                if question.lower() not in ['quit', 'q', 'exit']:
                    # Get the answer from the chain
                    hide_source = True
                    start = time.time()
                    if qa_type == "Conversational":
                        result = qa({
                            "question": question,
                            "chat_history": chat_history
                        })
                        answer = result['answer']
                        chat_history.append((question, answer))
                        answer, docs = answer, [] if hide_source else result['source_documents']
                    else:
                        result = qa(question)
                        answer, docs = result['result'], [] if hide_source else result['source_documents']
                    end = time.time()

                    # Print the result
                    with st.chat_message(name="user", avatar="🧑‍💻"):
                        st.write(question)

                    with st.chat_message(name="ai", avatar="🤖"):
                        st.write(answer.strip())
                        st.write(f"\n> Answer (took {round(end - start, 2)} seconds):")


def ui_spacer(n=2, line=False, next_n=0):
    for _ in range(n):
        st.write('')
    if line:
        st.tabs([' '])
    for _ in range(next_n):
        st.write('')


def ui_output():
    output = ss.get('output','')
    st.markdown(output)


def ui_show_debug():
    st.checkbox('show debug section', key='show_debug')


def ui_debug():
    if ss.get('show_debug'):
        st.write('### debug')
        st.write(ss.get('debug',{}))

def b_clear():
    if st.button('clear output'):
        ss['output'] = ''

if __name__ == "__main__":
    main()