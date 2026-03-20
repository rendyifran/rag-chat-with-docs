# RAG Chatbot for Personal Knowledge Base

<p align="center">
  <img src="images/RAG_Interface.jpg" width="30%" />
  <img src="images/RAG_Workflow.png" width="30%" />
  <img src="images/RAG_Performance.jpg" width="30%" />
</p>

## Overview

This project develops a custom Retrieval-Augmented Generation (RAG) assistant to support efficient querying and understanding of personal MSc and applied data science projects.

Using:
- Document-based knowledge (PDFs, reports, notes)
- ChromaDB (vector database)
- Ollama (LLM and embeddings)

The system demonstrates how combining retrieval, intent classification, and evaluation can improve the reliability of question-answering systems over personal knowledge bases.

---

## Objective

Design a chatbot that can accurately answer questions about project materials while handling variations in how users phrase their queries.

---

## Approach

### Intent-Aware Retrieval

Implemented a lightweight intent classification layer to categorise queries into:
- Definition
- Descriptive
- Causal
- Comparison

This enables the system to adapt retrieval and context selection strategies based on the type of question.

---

### Hybrid Retrieval Strategy

To improve robustness, the system combines:
- Semantic search using embeddings (Ollama)
- Keyword-based fallback for edge cases

This reduces failure cases where purely embedding-based retrieval may miss relevant information.

---

### Context Construction & Generation

Retrieved documents are filtered and structured using an intent-aware selection mechanism before being passed into the language model for answer generation.

The model produces:
- Grounded answers
- Source attribution for transparency

---

## Evaluation Framework

A custom evaluation pipeline was developed to assess performance across different query types.

Evaluation metrics include:
- Pass / Partial / Fail classification
- Coverage of expected answer components

### Results:
- Strong performance on definition queries (~88.9% pass rate)
- Moderate performance on descriptive queries
- Weak performance on causal and comparison queries

This highlights the gap between retrieval accuracy and reasoning capability.

---

## Chatbot Interface

A Streamlit-based interface provides an interactive way to query the system.

Features include:
- Natural language question input
- Display of generated answers
- Source document visibility
- Detected intent and response time tracking
- Chat history management

This allows practical usage as a personal knowledge assistant for revisiting past work.

---

## How to Run

### 1. Clone the repository

    git clone https://github.com/rendyifran/rag-chat-with-docs.git
    cd rag-chat-with-docs

### 2. Install dependencies

    pip install -r requirements.txt

### 3. Set up Ollama

Ensure Ollama is installed and running:

    ollama run llama3
    ollama pull nomic-embed-text

### 4. Create the vector database

    python create_database.py

### 5. Run evaluation (optional)

    python query_data.py --eval --eval_file eval_cases.json

Verbose mode:

    python query_data.py --eval --eval_file eval_cases.json --eval_verbose

### 6. Launch the chatbot

    streamlit run app.py

---

## Ethical & Methodological Considerations

- Personal document-based systems require careful handling of sensitive information
- Evaluation is essential to avoid overestimating system capability
- Retrieval-based systems may produce incomplete answers if relevant context is not retrieved
- Reasoning limitations remain a key challenge in RAG pipelines

---

## Tools

Python · Streamlit · ChromaDB · Ollama · LangChain · Vector Embeddings · NLP

---

## Repository Structure

    images/                – Visual outputs 
    Data/                  – Source documents  
    app.py                 – Streamlit chatbot interface  
    query_data.py          – RAG pipeline and query handling  
    create_database.py     – Document processing and embedding  
    eval_cases.json        – Evaluation dataset  
    requirements.txt       – Dependencies  
    README.md  

---

## Author

Mohamad Rendy Irawan Ifran  
MSc Social and Geographic Data Science  
University College London
