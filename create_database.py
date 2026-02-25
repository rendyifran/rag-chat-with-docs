from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader #loads multiple files from a folder and reads PDF files and extracts text
from langchain_text_splitters import RecursiveCharacterTextSplitter #splits long text into chunks
from langchain_core.documents import Document #represents a document with content and metadata, used for processing and storing text data in a structured format 
from langchain_ollama import OllamaEmbeddings #generates vector embeddings for text data using the Ollama API, allowing for efficient storage and retrieval of relevant documents based on their vector representations
from langchain_chroma import Chroma #stores and retrieves vector embeddings, used for efficient similarity search and retrieval of relevant documents based on their vector representations
from dotenv import load_dotenv #loads environment variables from a .env file, allowing you to keep sensitive information like API keys out of your codebase and easily manage them in a separate file.
import os
import shutil

load_dotenv() #loads API keys from .env


CHROMA_PATH = "chroma" #directory where the Chroma vector store will be saved, allowing for persistent storage of vector embeddings and efficient retrieval of relevant documents based on their vector representations
DATA_PATH = "Data" #directory where the PDF documents are stored, which will be loaded and processed to generate vector embeddings for semantic search and retrieval tasks


def main():
    generate_data_store()


def generate_data_store():
    documents = load_documents() #Load raw materials
    chunks = split_text(documents) #split the text into smaller pieces
    save_to_chroma(chunks) #Convert & store them in database


def load_documents():
    loader = DirectoryLoader(DATA_PATH, glob="**/*.pdf", loader_cls=PyPDFLoader) #loads all PDF files from the specified directory and its subdirectories, using the PyPDFLoader to extract text content from each PDF file and create Document objects for further processing
    documents = loader.load() #loads the documents from the specified directory, returning a list of Document objects that contain the extracted text content and associated metadata for each PDF file, which can then be processed and stored in a vector database for efficient retrieval and semantic search tasks.
    return documents


def split_text(documents: list[Document]): #uses the RecursiveCharacterTextSplitter to split the text content of each Document into smaller chunks.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len,
        add_start_index=True,
    ) #configures the text splitter to create chunks of up to 1000 characters with an overlap of 150 characters between consecutive chunks, using the built-in len function to determine the length of the text and adding a start index to each chunk for reference.
    chunks = text_splitter.split_documents(documents) 
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    if len(chunks) > 0:
        sample = chunks[min(10, len(chunks)-1)] #prints the content and metadata of a sample chunk for verification, showing the first 500 characters of the chunk's content and its associated metadata to confirm that the text splitting process is working as expected and that the resulting chunks contain the correct information.
        print(sample.page_content[:500])
        print(sample.metadata)
    print(sample.page_content)
    print(sample.metadata)

    return chunks


def save_to_chroma(chunks): 
    # Clear out the database first.
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # Use LOCAL embeddings (no OpenAI)
    embedding_function = OllamaEmbeddings(model="nomic-embed-text")

    # Create a new DB from the documents.
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_function,
        persist_directory=CHROMA_PATH
    )
    
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")


if __name__ == "__main__": #checks if the script is being run directly (as the main module) and calls the main() function to execute the data loading, text splitting, and storage process, allowing the script to be used as a standalone program for creating a Chroma vector store from PDF documents.
    main()