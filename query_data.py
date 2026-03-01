import argparse # Import the argparse module to handle command-line arguments for the script, allowing users to specify the query text when running the script from the command line
from langchain_chroma import Chroma # Import the Chroma class from the langchain_chroma module to create and manage a Chroma vector database for storing embeddings and associated metadata
from langchain_core.documents import Document # Import the Document class from the langchain_core.documents module to represent documents with content and metadata, which will be used for loading and processing the PDF documents before creating embeddings and storing in the Chroma vector database
from langchain_ollama import OllamaEmbeddings, ChatOllama # Import the OllamaEmbeddings and ChatOllama classes from the langchain_ollama module to generate embeddings using the Ollama model and to interact with the Ollama language model for generating responses based on the retrieved context from the Chroma vector database
from langchain_core.prompts import ChatPromptTemplate # Import the ChatPromptTemplate class from the langchain_core.prompts module to create a prompt template for formatting the context and question when generating a response using the Ollama language model

CHROMA_PATH = "chroma" # Directory where the Chroma vector database will be stored, allowing for persistence of the data store across runs of the script and enabling future queries to access the stored embeddings and associated metadata for efficient retrieval of relevant information based on similarity search

PROMPT = """ 
You must answer using ONLY the context below.

If the context does NOT explicitly contain the specific variables or indicators asked about, reply with exactly:
I don't know based on the provided documents.

If the context DOES explicitly list the variables or indicators, reply with ONE clear, natural sentence that states them.

Do NOT give vague or general answers.
Do NOT restate the question.
Do NOT guess."

Context:
{context}

Question: {question}

Answer in one clear, natural sentence using the context:
""" # Define a prompt template for generating responses using the Ollama language model, which instructs the model to answer based solely on the provided context and to indicate if the answer is not found in the context, ensuring that the responses are grounded in the retrieved information from the Chroma vector database

DEBUG = False


def main():
    # CLI
    parser = argparse.ArgumentParser() # Create an ArgumentParser object to handle command-line arguments for the script, allowing users to specify the query text when running the script from the command line
    parser.add_argument("query_text", type=str, help="The query text.") # Add a required positional argument "query_text" to the argument parser, which will be used to specify the query text that the user wants to ask based on the context retrieved from the Chroma vector database, allowing for dynamic querying of the data store when running the script from the command line
    args = parser.parse_args() # Parse the command-line arguments and store them in the args variable, which will be used to access the query text specified by the user when running the script from the command line
    query_text = args.query_text # Extract the query text from the parsed command-line arguments and store it in the query_text variable, which will be used for retrieving relevant context from the Chroma vector database and generating a response using the Ollama language model based on the specified query

    # Load vector DB (must match embedding model used when building DB)
    embedding_function = OllamaEmbeddings(model="nomic-embed-text") # Initialize the OllamaEmbeddings with the specified model ("nomic-embed-text") to create embeddings for the chunks of text that will be stored in the Chroma vector database, allowing for efficient similarity search and retrieval based on the content of the chunks. This embedding function must match the one used when building the database to ensure that the embeddings are compatible for similarity search and retrieval.
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function) # Create a Chroma vector database instance by specifying the directory where the Chroma vector database is stored (CHROMA_PATH) and the embedding function to use for creating embeddings (embedding_function), allowing for loading the existing data store and enabling similarity search and retrieval based on the content of the chunks stored in the Chroma vector database when processing the specified query text

    # Retrieve
    results = db.similarity_search_with_relevance_scores(query_text, k=50) # Perform a similarity search on the Chroma vector database using the specified query text and retrieve the top 20 most relevant chunks of text along with their relevance scores, allowing for retrieving context that is most similar to the query text for generating a response using the Ollama language model based on the retrieved context from the Chroma vector database
    if not results:
        print("Unable to find matching results.")
        return

    # Filter out noisy chunks (optional, but keep it light)
    filtered = []
    for doc, score in results:
        text = doc.page_content.lower()
        if "references" in text and "vulnerability" not in text:
            continue
        if "doi.org" in text:
            continue
        filtered.append((doc, score))

    results = filtered[:15] if filtered else results[:15]

    if DEBUG:
        for doc, score in results:
            print("Score:", score, "| Source:", doc.metadata.get("source")) # Print the relevance score and source metadata of each retrieved chunk for debugging purposes, allowing for verification of the relevance of the retrieved context and the associated sources from the Chroma vector database when processing the specified query text

    # Build context
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results]) # Build the context text by concatenating the content of the retrieved chunks of text, separating them with a delimiter ("\n\n---\n\n") to clearly distinguish between different chunks of context when generating a response using the Ollama language model based on the retrieved context from the Chroma vector database for the specified query text
    context_text = context_text[:3500] 

    # Build prompt (IMPORTANT: define prompt_template BEFORE using it)
    prompt_template = ChatPromptTemplate.from_template(PROMPT)
    prompt = prompt_template.format(context=context_text, question=query_text)

    if DEBUG: # If debugging is enabled, print the generated prompt to verify that the context and question are correctly formatted and included in the prompt for generating a response using the Ollama language model based on the retrieved context from the Chroma vector database for the specified query text
        print("\n--- PROMPT ---\n")
        print(prompt)
        print("\n--- END PROMPT ---\n")

    # LLM
    model = ChatOllama(model="llama3.2:3b", temperature=0)
    response = model.invoke(prompt)

    # Output
    sources = [doc.metadata.get("source", None) for doc, _score in results]
    print("\nResponse:\n", response.content)
    print("\nSources:", sources)


if __name__ == "__main__":
    main()

#python query_data.py "The Physical Infrastructure Vulnerability Index was based on what variables?"