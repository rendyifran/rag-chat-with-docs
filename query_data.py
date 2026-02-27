import argparse
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate

CHROMA_PATH = "chroma"

PROMPT = """
You must answer using ONLY the context below.
If the answer is not in the context, say: "Not found in the provided documents."

Context:
{context}

Question: {question}

Answer (one short sentence):
"""

DEBUG = False


def main():
    # CLI
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text

    # Load vector DB (must match embedding model used when building DB)
    embedding_function = OllamaEmbeddings(model="nomic-embed-text")
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Retrieve
    results = db.similarity_search_with_relevance_scores(query_text, k=20)
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

    results = filtered[:5] if filtered else results[:5]

    if DEBUG:
        for doc, score in results:
            print("Score:", score, "| Source:", doc.metadata.get("source"))

    # Build context
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

    # Build prompt (IMPORTANT: define prompt_template BEFORE using it)
    prompt_template = ChatPromptTemplate.from_template(PROMPT)
    prompt = prompt_template.format(context=context_text, question=query_text)

    if DEBUG:
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