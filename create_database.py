from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader # For loading PDF documents from a directory
from langchain_text_splitters import RecursiveCharacterTextSplitter # For splitting text into manageable chunks
from langchain_core.documents import Document # For representing documents with content and metadata
from langchain_ollama import OllamaEmbeddings # For generating embeddings using the Ollama model
from langchain_chroma import Chroma # For creating and managing a Chroma vector database
from dotenv import load_dotenv # For loading environment variables from a .env file
import os # For file system operations, such as checking if a directory exists and removing it
import shutil # For high-level file operations, such as removing a directory and its contents

load_dotenv() # Load environment variables from a .env file, if it exists


CHROMA_PATH = "chroma" # Directory where the Chroma vector database will be stored
DATA_PATH = "Data" # Directory where the PDF documents are located


def main():
    generate_data_store() # Generate the data store by loading documents, splitting them into chunks, and saving to Chroma


def generate_data_store(): # Main function to generate the data store by loading documents, splitting them into chunks, and saving to Chroma
    documents = load_documents() # Load PDF documents from the specified directory
    chunks = split_text(documents) # Split the loaded documents into manageable chunks using the RecursiveCharacterTextSplitter
    save_to_chroma(chunks) # Save the generated chunks to a Chroma vector database, creating embeddings using the Ollama model


def load_documents():
    loader = DirectoryLoader(DATA_PATH, glob="**/*.pdf", loader_cls=PyPDFLoader) # Create a DirectoryLoader to load PDF documents from the specified directory, using PyPDFLoader to handle PDF files
    documents = loader.load() # Load the documents from the directory and return them as a list of Document objects, each containing the content and metadata of a PDF document
    return documents # Return the loaded documents to be processed further in the data store generation process


def split_text(documents: list[Document]): # Split the loaded documents into manageable chunks using the RecursiveCharacterTextSplitter, which allows for overlapping chunks to preserve context
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=300,
        length_function=len,
        add_start_index=True,
    ) # Initialize the RecursiveCharacterTextSplitter with a chunk size of 800 characters, an overlap of 200 characters between chunks to preserve context, and a length function that uses the built-in len() function to determine the length of the text
    chunks = text_splitter.split_documents(documents) # Use the text splitter to split the loaded documents into chunks, which will be used for creating embeddings and storing in the Chroma vector database
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.") # Print the number of documents that were split and the total number of chunks generated for verification and debugging purposes

    if len(chunks) > 0: # If there are any chunks generated, print a sample chunk's content and metadata for verification and debugging purposes
        sample = chunks[min(10, len(chunks)-1)] # Get a sample chunk (the 10th chunk or the last chunk if there are fewer than 10) to display its content and metadata
        print(sample.page_content[:500]) # Print the first 500 characters of the sample chunk's content to verify that the text splitting is working correctly and to get a sense of the content being processed
        print(sample.metadata) # Print the metadata of the sample chunk to verify that the metadata is being preserved correctly during the text splitting process
    print(sample.page_content) # Print the entire content of the sample chunk for further verification and debugging purposes, allowing for a more comprehensive review of the text splitting results and the content being processed for embedding and storage in the Chroma vector database
    print(sample.metadata) # Print the metadata of the sample chunk again for further verification and debugging purposes, ensuring that the metadata is consistent and correctly associated with the chunk of text being processed for embedding and storage in the Chroma vector database

    return chunks


def save_to_chroma(chunks): # Save the generated chunks to a Chroma vector database, creating embeddings using the Ollama model. If the Chroma directory already exists, it will be removed to ensure a fresh start for the new data store.
    
    if os.path.exists(CHROMA_PATH): # Check if the Chroma directory already exists, which would indicate that there is an existing vector database that needs to be removed before creating a new one with the updated chunks
        shutil.rmtree(CHROMA_PATH) # Remove the existing Chroma directory and all its contents to ensure that the new data store is created from scratch with the updated chunks, preventing any potential conflicts or issues with leftover data from previous runs of the script

    
    embedding_function = OllamaEmbeddings(model="nomic-embed-text") # Initialize the OllamaEmbeddings with the specified model ("nomic-embed-text") to create embeddings for the chunks of text that will be stored in the Chroma vector database, allowing for efficient similarity search and retrieval based on the content of the chunks

    
    db = Chroma.from_documents(
        documents=chunks, # Create a Chroma vector database from the generated chunks of text, using the specified embedding function to create embeddings for each chunk and storing the resulting vector representations in the Chroma directory for efficient similarity search and retrieval in future queries
        embedding=embedding_function, # Use the initialized OllamaEmbeddings as the embedding function to create vector representations of the chunks of text, which will be stored in the Chroma vector database for efficient similarity search and retrieval based on the content of the chunks
        persist_directory=CHROMA_PATH # Specify the directory where the Chroma vector database will be stored, allowing for persistence of the data store across runs of the script and enabling future queries to access the stored embeddings and associated metadata for efficient retrieval of relevant information based on similarity search
    )
    
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.") # Print the number of chunks that were saved to the Chroma vector database and the directory where it was stored for verification and debugging purposes, confirming that the data store was successfully created with the expected number of chunks and is available for future queries and retrieval based on similarity search


if __name__ == "__main__": # Check if the script is being run directly (as the main module) and if so, execute the main() function to start the process of generating the data store by loading documents, splitting them into chunks, and saving to Chroma
    main()