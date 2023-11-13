import os
import glob
import re
from common import config
from typing import List
from multiprocessing.pool import ThreadPool
from tqdm import tqdm

# Rich
from rich.console import Console
console = Console()

from langchain.document_loaders import (
    CSVLoader,
    EverNoteLoader,
    PyMuPDFLoader,
    TextLoader,
    UnstructuredEPubLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredODTLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
)

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

# Map file extensions to document loaders and their arguments
LOADER_MAPPING = {
    ".csv": (CSVLoader, {}),
    # ".docx": (Docx2txtLoader, {}),
    ".doc": (UnstructuredWordDocumentLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".enex": (EverNoteLoader, {}),
    ".epub": (UnstructuredEPubLoader, {}),
    ".html": (UnstructuredHTMLLoader, {}),
    ".md": (UnstructuredMarkdownLoader, {}),
    ".odt": (UnstructuredODTLoader, {}),
    ".pdf": (PyMuPDFLoader, {}),
    ".ppt": (UnstructuredPowerPointLoader, {}),
    ".pptx": (UnstructuredPowerPointLoader, {}),
    ".txt": (TextLoader, {}),

    # Programming languages
    ".ts": (TextLoader, {}),
    ".js": (TextLoader, {}),
    ".css": (TextLoader, {}),
    ".py": (TextLoader, {}),
    ".php": (TextLoader, {}),
    ".go": (TextLoader, {}),
    ".mod": (TextLoader, {}),
    ".java": (TextLoader, {}),
    ".kt": (TextLoader, {}),
    ".rb": (TextLoader, {}),
    ".json": (TextLoader, {}),
    ".gitignore": (TextLoader, {}),
    ".dockerfile": (TextLoader, {}),
    ".sql": (TextLoader, {}),
    ".cpp": (TextLoader, {}),
    ".sh": (TextLoader, {}),
    ".jsx": (TextLoader, {}),
    ".scss": (TextLoader, {}),
    ".less": (TextLoader, {}),
    ".svg": (TextLoader, {}),

    ".yaml": (TextLoader, {}),
    ".xml": (TextLoader, {}),
    ".xls": (TextLoader, {}),
    ".xlsx": (TextLoader, {}),
    ".toml": (TextLoader, {}),
    ".ini": (TextLoader, {}),
    ".yml": (TextLoader, {}),
}

def remove_comments(text):
    for x in re.findall(r'("[^\n]*"(?!\\))|(//[^\n]*$|/(?!\\)\*[\s\S]*?\*(?!\\)/)', text, 8):
        print(x[1])
        text = text.replace(x[1], '')

    text = re.sub(r'(?m) *#.*\n?', '', text)
    return text

def load_single_document(file_path: str) -> List[Document]:
    ext = os.path.splitext(file_path)[-1]
    if ext in LOADER_MAPPING:
        try:
            loader_class, loader_args = LOADER_MAPPING[ext]
            loader = loader_class(file_path, **loader_args)
            return loader.load()
        except Exception as e:
            return {'exception': e, 'file': file_path}

    raise ValueError(f"Unsupported file extension '{ext}'")


def load_documents(source_dir: str, ignored_files: List[str] = []) -> List[Document]:
    """
    Loads all documents from the source documents directory, ignoring specified files
    """
    all_files = []
    for ext in LOADER_MAPPING:
        all_files.extend(
            glob.glob(os.path.join(source_dir, f"**/*{ext.lower()}"), recursive=True)
        )
        all_files.extend(
            glob.glob(os.path.join(source_dir, f"**/*{ext.upper()}"), recursive=True)
        )
    filtered_files = [file_path for file_path in all_files if file_path not in ignored_files]

    with ThreadPool(processes=os.cpu_count()) as pool:
        results = []
        with tqdm(total=len(filtered_files), desc='Loading new documents', ncols=80) as pbar:
            for i, docs in enumerate(pool.imap_unordered(load_single_document, filtered_files)):
                if isinstance(docs, dict):
                    console.print(" - " + docs['file'] + ": ERROR: " + str(docs['exception']), style="red")
                    continue
                results.extend(docs)
                pbar.update()

    return results


def process_documents(source_directory: str, ignored_files: List[str] = []) -> List[Document]:
    """
    Load documents and split in chunks
    """
    console.print(f" Loading documents from {source_directory}", style="yellow")
    documents = load_documents(source_directory, ignored_files)
    if not documents:
        console.print("No new documents to load. If this problem persists, please try removing the repository directory and vector database manually.", style="red")
        exit(1)

    console.print(f"Loaded {len(documents)} new documents from {source_directory}", style="yellow")

    # Strip all comments from the files (Don't know why this is not working, please fix :).
    if config.get('REMOVE_COMMENTS'):
        for i in range(len(documents)):
            documents[i].page_content = remove_comments(documents[i].page_content)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.get('CHUNK_SIZE'),
        chunk_overlap=config.get('CHUNK_OVERLAP')
    )

    documents = text_splitter.split_documents(documents)
    console.print(f"Split into {len(documents)} chunks of text (max. {config.get('CHUNK_SIZE')} tokens each with an overlap of {config.get('CHUNK_OVERLAP')})", style="yellow")

    return documents


def split_list(list_object, chunk_size):
    chunks = []
    for start in range(0, len(list_object), chunk_size):
        stop = start + chunk_size
        chunks.append(list_object[start:stop])
    return chunks

