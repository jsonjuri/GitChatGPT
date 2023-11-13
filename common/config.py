import os
from dotenv import load_dotenv
import textwrap

# Rich
from rich.console import Console
console = Console()

def load_prompt():
    with open('./prompt.txt', 'r') as file:
        text_content = file.read()
    return text_content

prompt_template = textwrap.dedent(load_prompt())

# Define a list of providers that should be treated as local servers.
local_servers = [
    'openai',
    'localhost',
    'huggingface',
    'ollama',
    'openlm',
    'openllm',
    'chatglm',
    'xinference'
]

defaults = {
    'VERBOSE': False,
    'PROMPT_ENABLED': False,
    'GPU_ENABLED': False,
    'CPU_PERCENTAGE': 0.8,
    'DEFAULT_LLM': 'llamacpp',
    'DEFAULT_VECTOR_DB': 'chroma',
    'DOCUMENT_PATH': "./documents",
    'DB_PATH': "./db",
    'VECTOR_DB': "chroma",
    'DB_TELEMETRY': False,
    'EMBEDDINGS_MODEL_NAME': "all-MiniLM-L6-v2",
    'CHUNK_SIZE': 500,
    'CHUNK_OVERLAP': 50,
    'TARGET_SOURCE_CHUNKS': 4,
    'REMOVE_COMMENTS': True,
    'MODEL_TYPE': "openai",
    'MODEL_PATH': "./models/",
    'MODEL': "ggml-gpt4all-j-v1.3-groovy.bin",
    'MODEL_REVISION': 'v1.3-groovy',
    'MODEL_MAX_TOKENS': 2048,
    'MODEL_N_GPU_LAYERS': 4,
    'MODEL_N_BATCH': 1024,
    'MODEL_USE_MLOCK': True,
    'MODEL_TEMPERATURE': 0.75,
    'MODEL_STREAMING': False,
    'MODEL_TOP_K': 40,
    'MODEL_TOP_P': 0.9,
    'OLLAMA_BASE_URL': 'http://127.0.0.1:5001',
    'LOCALHOST_SERVER': 'http://127.0.0.1:5001',
    'LOCALHOST_API_VERSION': 'v1',
    'LOCALHOST_STREAMING': False,
    'OPENAI_API_KEY': 'sk-111111111111111111111111111111111111111111111111',
    'OPENAI_ORGANIZATION': '',
    'OPENAI_STREAMING': False,
    'OPENAI_EMBEDDINGS': False,
    'GITHUB_REPOSITORY': 'https://github.com/jsonjuri/gitchatgpt.git',
    'GITHUB_USERNAME': '',
    'GITHUB_ACCESS_TOKEN': ''
}

def get_list(list_type: str):
    # Read the array from the environment variable
    item_list = os.environ.get(list_type)

    # Split the string into a list
    if ',' in item_list:
        return item_list.split(',')
    else:
        return [
            item_list
        ]

def get_list_selected(list_type: str, selected: int):
    item_list = get_list(list_type)
    return str(item_list[selected]).lower()

def get(key: str):
    if key not in defaults:
        print(f"{key} is missing in the default object in config.py")

    default_value = defaults.get(key)
    value = os.environ.get(key, default_value)

    # A dirty hack to make sure TRUE will become True and FALSE will become False
    # Without using this, it may not be recognized as a boolean.
    if str(value).lower() == 'true':
        return True
    elif str(value).lower() == 'false':
        return False
    elif str(value).isnumeric():
        if isinstance(value, float):
            return int(value)
        else:
            return int(value)
    else:
        return value


def display_intro():
    message = """
_____________________________________________________________________________________________________________ 


         ██████╗ ██╗████████╗     ██████╗██╗  ██╗ █████╗ ████████╗     ██████╗ ██████╗ ████████╗
        ██╔════╝ ██║╚══██╔══╝    ██╔════╝██║  ██║██╔══██╗╚══██╔══╝    ██╔════╝ ██╔══██╗╚══██╔══╝
        ██║  ███╗██║   ██║       ██║     ███████║███████║   ██║       ██║  ███╗██████╔╝   ██║   
        ██║   ██║██║   ██║       ██║     ██╔══██║██╔══██║   ██║       ██║   ██║██╔═══╝    ██║   
        ╚██████╔╝██║   ██║       ╚██████╗██║  ██║██║  ██║   ██║       ╚██████╔╝██║        ██║   
         ╚═════╝ ╚═╝   ╚═╝        ╚═════╝╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝        ╚═════╝ ╚═╝        ╚═╝                                                                                        

_____________________________________________________________________________________________________________ 
    """
    console.print(message, style="#e8e4ca bold")

    console.print(
        """GitChatGPT is a tool that fetches GitHub repositories or scans through documents, then zaps them into 
a Language Model (LLM) of your choosing. Now, you're the question maestro—ask away about the repository 
or the docs you dropped in! 🤖🌐

Tools and libraries used: 
    * [bold]Langchain 🦜[/bold] to build and compose LLMs
    * [bold]ChromaDB, DeepLake[/bold] to store vectors (a.k.a [italic]embeddings[/italic]).
    * [bold]Rich[/bold] to build a cool terminal UX/UI

LLM Options:   
    * [bold]Localhost Server[/bold] (For example with LMStudio)
    * [bold]LlamaCpp[/bold]
    * [bold]GPT4ALL[/bold]
    * [bold]Ollama[/bold]
    * [bold]ChatGLM[/bold]
    * [bold]OpenLLM[/bold]
    * [bold]OpenLM[/bold]
    * [bold]XInference[/bold]
    * [bold]Huggingface[/bold]
    * [bold]OpenAI[/bold] (🔑 needed)
_____________________________________________________________________________________________________________ 

Let's start :rocket:
    """, style="#e8e4ca")

def switch_dotenv(model_choice: any):
    try:
        if str(model_choice).isnumeric():
            model_choice = int(model_choice)
            dotenv_file = get_list_selected('LLM_PROVIDERS', model_choice)
        else:
            dotenv_file = model_choice

        if dotenv_file:
            filename = dotenv_file.lower()
            if ' ' not in filename:
                env_path = os.path.join("env/", filename)
                env_file = f"{env_path}.env"
                if not load_dotenv(dotenv_path=env_file):
                    console.print(f"Could not load the .env file from {env_file}, or it is empty. Please check if it exists and is readable.", style="red")
                    exit(1)

                # Get the model name (without .env).
                model_name_display = dotenv_file.replace(".env", "")

                # Set the MODEL_TYPE variable
                os.environ['MODEL_TYPE'] = model_name_display

                if str(model_choice).isnumeric():
                    console.print(f"{model_name_display} has been set as your model llm.", style="green")
                    print("")
        else:
            console.print(f"Env file '{dotenv_file}' not found.", style="red")
            exit(1)
    except (ValueError, IndexError):
        console.print("Invalid choice. Please enter a valid index.", style="red")
        exit(1)

def check_directories():
    # Make sure the database directory exists; otherwise, create it.
    if not os.path.exists(get('DB_PATH')):
        console.print(f"Creating a directory for storing the database: {get('DB_PATH')}", style="cyan")
        os.mkdir(get('DB_PATH'))

    # Make sure the documents directory exists; otherwise, create it.
    if not os.path.exists(get('DOCUMENT_PATH')):
        console.print(f"Creating a directory for storing documents: {get('DOCUMENT_PATH')}", style="cyan")
        os.mkdir(get('DOCUMENT_PATH'))

    # Make sure the model llm directory exists; otherwise, create it.
    if not os.path.exists(get('MODEL_PATH')):
        console.print(f"Creating a directory for storing the model llm: {get('MODEL_PATH')}", style="cyan")
        os.mkdir(get('MODEL_PATH'))