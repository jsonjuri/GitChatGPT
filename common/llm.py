import os
from common import config
from common import gpu

from langchain.cache import InMemoryCache
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import (
    GPT4All,
    LlamaCpp,
    Ollama,
    OpenAI,
    OpenLM,
    OpenLLM,
    Xinference,
    ChatGLM,
    HuggingFacePipeline
)

from langchain.embeddings import (
    OpenAIEmbeddings,
    OllamaEmbeddings,
    SentenceTransformerEmbeddings,
    BedrockEmbeddings,
    HuggingFaceEmbeddings,
)

from transformers import AutoTokenizer, pipeline, AutoModelForSeq2SeqLM

# Rich
from rich.console import Console
console = Console()

def load_embedding_model(embedding_model_name: str, settings=None):
    if settings is None:
        settings = {}

    match embedding_model_name:
        case "ollama":
            embeddings = OllamaEmbeddings(
                base_url=settings["ollama_base_url"],
                model="llama2"
            )

            dimension = 4096
            console.print("Embedding: Using Ollama")
        case "openai":
            embeddings = OpenAIEmbeddings()
            dimension = 1536

            console.print("Embedding: Using OpenAI")
        case "aws":
            embeddings = BedrockEmbeddings()
            dimension = 1536

            console.print("Embedding: Using AWS")
        case _default:
            embeddings = SentenceTransformerEmbeddings(
                model_name="all-MiniLM-L6-v2",
                cache_folder="/"
            )
            dimension = 384

            console.print("Embedding: Using SentenceTransformer")

    return (
        embeddings,
        dimension
    )

def load_llm(llm_name: str, model: str, model_path: str, stream: bool):
    callbacks = [] if stream else [StreamingStdOutCallbackHandler()]

    # Get the number of threads that will be used.
    model_n_threads = int(os.cpu_count() * float(config.get('CPU_PERCENTAGE'))) if config.get('CPU_PERCENTAGE') else 4

    # Set the GPU layers based on config file (0 = automatically)
    n_gpu_layers = gpu.calculate_layer_count() if config.get('MODEL_N_GPU_LAYERS') == 0 else config.get('MODEL_N_GPU_LAYERS')

    # Cache
    # langchain.llm_cache = InMemoryCache()

    console.print("Chill out! 😎 We're getting the model ready for you. Just a couple of minutes... ⏳", style="yellow")

    # Prepare the Language Model (LLM)
    match llm_name:
        case "llamacpp":
            llm = LlamaCpp(
                model_path=model_path,
                temperature=config.get('MODEL_TEMPERATURE'),
                max_tokens=config.get('MODEL_MAX_TOKENS'),
                n_ctx=config.get('MODEL_MAX_TOKENS'),
                n_batch=config.get('MODEL_N_BATCH'),
                f16_kv=True,
                callbacks=callbacks,
                verbose=config.get('VERBOSE'),
                n_gpu_layers=n_gpu_layers,
                n_threads=model_n_threads,
                use_mlock=config.get('MODEL_USE_MLOCK'),
                top_p=config.get('MODEL_TOP_P'),
            )

        case "gpt4all":
            # gpt4all is broken after upgrading langchain.
            # If you are planning to fix this, make sure you add gpt4all to common/config.py in the model_llms array.
            # Please don't forget ask a pull-request after fixing this issue.
            llm = GPT4All(
                model=os.path.join(f".{os.path.sep}", model_path),
                backend="gptj",
                callbacks=callbacks,
                n_threads=model_n_threads,
                max_tokens=config.get('MODEL_MAX_TOKENS'),
                temp=config.get('MODEL_TEMPERATURE'),
                top_k=config.get('MODEL_TOP_K'),
                top_p=config.get('MODEL_TOP_P'),
                device='gpu' if config.get('GPU_ENABLED') else 'cpu',
                use_mlock=config.get('MODEL_USE_MLOCK'),
                streaming=config.get('MODEL_STREAMING'),
                verbose=config.get('VERBOSE')
            )

        case "ollama":
            llm = Ollama(
                base_url=config.get('OLLAMA_BASE_URL'),
                model=model,
                verbose=config.get('VERBOSE'),
                num_ctx=config.get('MODEL_MAX_TOKENS'),
                num_thread=model_n_threads,
                temperature=config.get('MODEL_TEMPERATURE'),
                top_k=config.get('MODEL_TOP_K'),
                top_p=config.get('MODEL_TOP_P'),
                callbacks=callbacks,
            )

        case "huggingface":
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            pretrained_model = AutoModelForSeq2SeqLM.from_pretrained(
                model_path,
                device_map='auto'
            )

            pipe = pipeline(
                "text2text-generation",
                model=pretrained_model,
                tokenizer=tokenizer
            )

            llm = HuggingFacePipeline(
                pipeline=pipe
            )

        case "localhost":
            # Common error: AttributeError: module 'openai' has no attribute 'Completion'. Did you mean: 'completions'?
            # Fix https://github.com/langchain-ai/langchain/issues/12967
            # pip install --upgrade openai==0.28.1
            os.environ['OPENAI_API_KEY'] = '[REDACTED]'
            os.environ['OPENAI_ORGANIZATION'] = '[REDACTED]'

            llm = OpenAI(
                model=model_path,
                openai_api_key=os.environ.get('OPENAI_API_KEY'),
                openai_organization=os.environ.get('OPENAI_ORGANIZATION'),
                streaming=config.get('LOCALHOST_STREAMING'),
                verbose=config.get('VERBOSE'),
            )

            import openai
            openai.api_base = f"{config.get('LOCALHOST_SERVER')}/{config.get('LOCALHOST_API_VERSION')}"

        case "chatglm":
            llm = ChatGLM(
                endpoint_url=config.get('LOCALHOST_SERVER'),
                max_token=config.get('MODEL_MAX_TOKENS'),
                history=[["我将从美国到中国来旅游，出行前希望了解中国的城市", "欢迎问我任何问题。"]],
                top_p=config.get('MODEL_TOP_P'),
                model_kwargs={"sample_model_args": False},
            )

        case "openlm":
            llm = OpenLM(
                model=model
            )

        case "openllm":
            llm = OpenLLM(
                model_name=model,
                model_id=config.get('MODEL_ID'),
                temperature=config.get('MODEL_TEMPERATURE'),
                repetition_penalty=config.get('MODEL_TOP_P'),
            )

        case "xinference":
            llm = Xinference(
                server_url=config.get('LOCALHOST_SERVER'),
                model_uid=model
            )

        case "openai":
            llm = OpenAI(
                # model=model,
                openai_api_key=config.get('OPENAI_API_KEY'),
                openai_organization=os.environ.get('OPENAI_ORGANIZATION'),
                streaming=config.get('OPENAI_STREAMING'),
                max_tokens=config.get('MODEL_MAX_TOKENS'),
                verbose=config.get('VERBOSE')
            )

        case _default:
            # Raise an exception if the model_type is not supported
            console.print(f"Model type {config.get('MODEL_TYPE')} is not supported.", style="red")
            exit(1)

    return llm

def set_llm(model, model_type, model_path):
    llm_name = str(model_type).lower()
    local_model = True
    if llm_name not in config.local_servers and all(
            config.get(key) for key in ['MODEL_TYPE', 'MODEL_PATH', 'MODEL']):
        if "." not in config.get('MODEL'):
            console.print("The MODEL_PATH in your env file must contain a file extension.", style="red")
            exit(1)

        model_base_path = os.path.join(model_path, llm_name)
        model_path = os.path.join(model_base_path, model)
        if not os.path.exists(model_path):
            console.print(
                f"Model not found, please make sure you copied {model} to {model_base_path}{os.path.sep}",
                style="red")
            exit(1)
    elif llm_name in config.local_servers:
        model_path = model
        local_model = False
    else:
        console.print("Please set the MODEL, MODEL_PATH, and MODEL_TYPE environment variables.", style="red")
        exit(1)

    return (
        llm_name,
        model_path,
        local_model
    )

def extract_title_and_question(input_string):
    lines = input_string.strip().split("\n")

    title = ""
    question = ""
    is_question = False  # flag to know if we are inside a "Question" block

    for line in lines:
        if line.startswith("Title:"):
            title = line.split("Title: ", 1)[1].strip()
        elif line.startswith("Question:"):
            question = line.split("Question: ", 1)[1].strip()
            is_question = (
                True  # set the flag to True once we encounter a "Question:" line
            )
        elif is_question:
            # if the line does not start with "Question:" but we are inside a "Question" block,
            # then it is a continuation of the question
            question += "\n" + line.strip()

    return title, question