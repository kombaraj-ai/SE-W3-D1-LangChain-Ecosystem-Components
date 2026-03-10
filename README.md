# Week 3 -> Day 1 -> LangChain Ecosystem and Components
---

## Table of Contents

1. [What is LangChain?](#1-what-is-langchain)
2. [LangChain Ecosystem](#2-langchain-ecosystem)
3. [LangChain Components](#3-langchain-components)
   - [Data Connection](#31-data-connection)
   - [Model I/O](#32-model-io)
   - [Chains](#33-chains)
   - [Memory](#34-memory)
   - [Agents](#35-agents)
   - [Tools](#36-tools)
4. [Hello World — LangChain Application](#4-hello-world--langchain-application)
5. [LangChain General Workflow](#langchain-general-workflow)
6. [Assignment](#assignment)
---

## 1. What is LangChain?

**LangChain** is an open-source framework designed to simplify the development of applications powered by **Large Language Models (LLMs)**. It provides a standard interface and a rich set of abstractions to compose LLMs with other components such as memory, data sources, APIs, and tools — enabling you to build complex, production-ready AI pipelines.

### Why LangChain?

Building LLM applications from scratch is hard. You need to:
- Manage prompts and their templates
- Connect LLMs to external data (PDFs, databases, APIs)
- Store and retrieve conversation history
- Enable LLMs to use tools (search, calculators, etc.)
- Chain multiple AI steps together

LangChain solves all of this with **reusable, composable building blocks**.

### Key Characteristics

| Feature | Description |
|---|---|
| **Model Agnostic** | Works with OpenAI, Anthropic, Ollama, HuggingFace, and many more |
| **Composable** | Build complex pipelines by chaining simple components |
| **Extensible** | Easily add custom components or integrations |
| **Production Ready** | LangSmith for tracing, LangServe for deployment |
| **Open Source** | Apache 2.0 licensed, large community |

### Core Philosophy

LangChain is built around the concept of **chains** — sequences of calls to LLMs, tools, or data transformations. These chains can be simple (one LLM call) or complex (multi-step reasoning with tool use and memory).

```
User Input → Prompt Template → LLM → Output Parser → Final Response
```

---

## 2. LangChain Ecosystem

The LangChain ecosystem is modular. It is split into multiple packages so you only install what you need.

```
┌─────────────────────────────────────────────────────────────┐
│                     LangChain Ecosystem                     │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────┐  │
│  │langchain-core│  │  langchain   │  │langchain-community│  │
│  │  (base ABCs) │  │ (chains,etc) │  │  (integrations)   │  │
│  └──────────────┘  └──────────────┘  └───────────────────┘  │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │  LangSmith   │  │  LangServe   │  │    Templates     │   │
│  │(observability│  │ (deployment) │  │ (starter apps)   │   │
│  └──────────────┘  └──────────────┘  └──────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

### 2.1 `langchain-core`

**Package:** `langchain-core`  
**Purpose:** The foundation of the entire LangChain framework.

`langchain-core` contains all the **base abstractions and interfaces** that every other LangChain package builds upon. It has **no heavy dependencies** and is extremely lightweight.

**What it includes:**
- Base classes for LLMs, Chat Models, Embeddings
- `Runnable` interface — the backbone of LCEL (LangChain Expression Language)
- `BasePromptTemplate`, `BaseChatPromptTemplate`
- `BaseOutputParser`
- `BaseMemory`
- `BaseRetriever`
- `Document` data class
- `RunnablePassthrough`, `RunnableLambda`, `RunnableParallel` for pipeline composition

**Example:**
```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
```

> 💡 **Key Concept:** The `Runnable` interface means every component has `.invoke()`, `.stream()`, `.batch()`, and `.ainvoke()` methods — enabling consistent composition.

---

### 2.2 `langchain-community`

**Package:** `langchain-community`  
**Purpose:** Third-party integrations contributed by the community.

This package contains integrations with external services, tools, and providers. It's where you find connectors for databases, vector stores, document loaders, and LLM providers not maintained by the core LangChain team.

**What it includes:**
- **Document Loaders:** `PyPDFLoader`, `WebBaseLoader`, `CSVLoader`, `NotionDBLoader`, etc.
- **Vector Stores:** `FAISS`, `Chroma`, `Pinecone`, `Weaviate`, `Qdrant`, etc.
- **LLMs:** `Ollama`, `HuggingFaceHub`, `Cohere`, `AI21`, etc.
- **Chat Models:** `ChatOllama`, `ChatHuggingFace`, etc.
- **Embeddings:** `OllamaEmbeddings`, `HuggingFaceEmbeddings`, etc.
- **Tools:** `DuckDuckGoSearchRun`, `WikipediaQueryRun`, `ArxivQueryRun`, etc.
- **Memory:** `RedisChatMessageHistory`, `MongoDBChatMessageHistory`, etc.

**Example:**
```python
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
```

---

### 2.3 `langchain`

**Package:** `langchain`  
**Purpose:** The main package containing chains, agents, and high-level abstractions.

This is the primary package most users interact with. It depends on `langchain-core` and provides high-level constructs that combine multiple components together.

**What it includes:**
- **Chains:** `LLMChain`, `RetrievalQA`, `ConversationalRetrievalChain`, `SequentialChain`
- **Agents:** `AgentExecutor`, `create_react_agent`, `create_tool_calling_agent`
- **Memory:** `ConversationBufferMemory`, `ConversationSummaryMemory`, etc.
- **Text Splitters:** `RecursiveCharacterTextSplitter`, `TokenTextSplitter`
- **LCEL Helpers:** `RunnableWithMessageHistory`

**Example:**
```python
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
```

---

### 2.4 Templates

**Purpose:** Pre-built, reference LangChain applications that can be used as starters.

LangChain Templates are **deployable reference architectures** for common use cases. You can scaffold them using the LangChain CLI and customize them for your needs.

**Common Templates:**
| Template | Description |
|---|---|
| `rag-chroma` | RAG pipeline using Chroma vector store |
| `openai-functions-agent` | Agent using OpenAI function calling |
| `sql-research-assistant` | Agent for querying SQL databases |
| `rag-conversation` | Conversational RAG with memory |
| `hyde-rag` | Hypothetical Document Embeddings RAG |
| `llama2-functions` | LLaMA 2 with function calling |

**Using Templates:**
```bash
# Install LangChain CLI
pip install langchain-cli

# Create app from template
langchain app new my-app --package rag-chroma

# Serve the app
langchain serve
```

---

### 2.5 LangServe

**Package:** `langserve`  
**Purpose:** Deploy LangChain chains and agents as REST APIs instantly.

LangServe makes it trivial to expose any LangChain `Runnable` as a production REST API. It is built on top of **FastAPI** and **Pydantic**.

**What it provides:**
- Automatic REST API generation from any `Runnable`
- Built-in `/invoke`, `/batch`, `/stream`, `/stream_log` endpoints
- Auto-generated **Swagger UI** docs at `/docs`
- A built-in **Playground UI** at `/playground`
- Client SDK (`RemoteRunnable`) to call deployed chains from Python

**Server Example:**
```python
# server.py
from fastapi import FastAPI
from langserve import add_routes
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate

app = FastAPI(title="My LangChain App")

prompt = ChatPromptTemplate.from_template("Tell me a joke about {topic}")
model = Ollama(model="llama3.2")
chain = prompt | model

# Add routes to FastAPI
add_routes(app, chain, path="/joke")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

**Client Example:**
```python
from langserve import RemoteRunnable

chain = RemoteRunnable("http://localhost:8000/joke")
result = chain.invoke({"topic": "cats"})
```

---

### 2.6 LangSmith

**Service:** https://smith.langchain.com  
**Purpose:** Observability, debugging, testing, and monitoring for LangChain applications.

LangSmith is a **SaaS platform** (with self-hosting options) that provides end-to-end visibility into your LLM application. It is the production monitoring and evaluation tool of the LangChain ecosystem.

**Key Features:**
| Feature | Description |
|---|---|
| **Tracing** | Auto-captures every LLM call, chain step, tool use |
| **Debugging** | Inspect inputs/outputs at every step of a chain |
| **Datasets** | Build evaluation datasets from real traces |
| **Evaluations** | Run automated evals on your chains |
| **Monitoring** | Track latency, token usage, costs over time |
| **Hub** | Share and discover prompts (LangChain Hub) |

**Setup:**
```python
import os

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your-langsmith-api-key"
os.environ["LANGCHAIN_PROJECT"] = "my-langchain-project"

# All LangChain calls are now automatically traced!
```

---

## 3. LangChain Components

### 3.1 Data Connection

Data Connection is about getting **external data into your LLM pipeline**. It covers loading, transforming, embedding, storing, and retrieving documents.

```
Raw Data → Document Loaders → Text Splitters → Embeddings → Vector Stores → Retrievers → LLM
```

---

#### 3.1.1 Document Loaders

Document Loaders load data from **various sources** into a standard `Document` object with `.page_content` (text) and `.metadata` (dict).

**Built-in Loaders:**

| Loader | Source | Package |
|---|---|---|
| `TextLoader` | `.txt` files | `langchain-community` |
| `PyPDFLoader` | PDF files | `langchain-community` |
| `CSVLoader` | CSV files | `langchain-community` |
| `WebBaseLoader` | Web pages | `langchain-community` |
| `JSONLoader` | JSON files | `langchain-community` |
| `DirectoryLoader` | Entire folders | `langchain-community` |
| `NotionDBLoader` | Notion databases | `langchain-community` |
| `GitLoader` | Git repositories | `langchain-community` |
| `YoutubeLoader` | YouTube transcripts | `langchain-community` |
| `ArxivLoader` | ArXiv papers | `langchain-community` |
| `ConfluenceLoader` | Confluence pages | `langchain-community` |

**Example:**
```python
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader, TextLoader

# Load a PDF
loader = PyPDFLoader("my_document.pdf")
docs = loader.load()
print(docs[0].page_content)  # First page text
print(docs[0].metadata)      # {'source': 'my_document.pdf', 'page': 0}

# Load a web page
web_loader = WebBaseLoader("https://python.langchain.com/docs/introduction")
web_docs = web_loader.load()

# Load a directory of .txt files
from langchain_community.document_loaders import DirectoryLoader
dir_loader = DirectoryLoader("./docs/", glob="**/*.txt", loader_cls=TextLoader)
all_docs = dir_loader.load()
print(f"Loaded {len(all_docs)} documents")
```

---

#### 3.1.2 Document Transformers (Text Splitters)

LLMs have a **context window limit**. Document Transformers (primarily Text Splitters) break large documents into smaller, overlapping **chunks** that fit within the model's context.

**Why overlap?** To avoid losing context at chunk boundaries — if a sentence spans two chunks, the overlap ensures it appears in at least one complete chunk.

**Common Text Splitters:**

| Splitter | Strategy |
|---|---|
| `RecursiveCharacterTextSplitter` | Splits by `\n\n`, `\n`, ` `, `` in order — **recommended default** |
| `CharacterTextSplitter` | Splits by a single separator character |
| `TokenTextSplitter` | Splits by token count (model-aware) |
| `MarkdownHeaderTextSplitter` | Splits by Markdown headings |
| `HTMLHeaderTextSplitter` | Splits by HTML heading tags |
| `CodeTextSplitter` | Language-aware code splitting |
| `SentenceTransformersTokenTextSplitter` | Splits for sentence transformer models |

**Example:**
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,       # Max characters per chunk
    chunk_overlap=200,     # Characters to overlap between chunks
    length_function=len,   # Function to measure chunk length
    add_start_index=True,  # Add start position to metadata
)

# Split documents
chunks = text_splitter.split_documents(docs)
print(f"Split {len(docs)} documents into {len(chunks)} chunks")
print(f"Sample chunk:\n{chunks[0].page_content}")
print(f"Metadata: {chunks[0].metadata}")

# Split plain text
text = "Your very long text here..."
text_chunks = text_splitter.split_text(text)
```

---

#### 3.1.3 Embedding Models

Embeddings convert text into **numerical vectors** (arrays of floats) that capture the semantic meaning of the text. Similar texts produce vectors that are close together in vector space, enabling **semantic search**.

```
"What is Python?" → [0.023, -0.114, 0.987, 0.345, ...]  (1536 dimensions)
"Python programming language" → [0.025, -0.110, 0.981, 0.340, ...]  (similar!)
```

**Common Embedding Models:**

| Embedding | Provider | Package |
|---|---|---|
| `OpenAIEmbeddings` | OpenAI (`text-embedding-3-small`) | `langchain-openai` |
| `OllamaEmbeddings` | Ollama (local) | `langchain-community` |
| `HuggingFaceEmbeddings` | HuggingFace models | `langchain-community` |
| `CohereEmbeddings` | Cohere | `langchain-community` |
| `BedrockEmbeddings` | AWS Bedrock | `langchain-community` |
| `GoogleGenerativeAIEmbeddings` | Google | `langchain-google-genai` |

**Example:**
```python
from langchain_community.embeddings import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Embed a single query
query_vector = embeddings.embed_query("What is LangChain?")
print(f"Embedding dimension: {len(query_vector)}")

# Embed multiple documents
doc_vectors = embeddings.embed_documents([
    "LangChain is an AI framework.",
    "Python is a programming language.",
    "Cats are mammals."
])
print(f"Embedded {len(doc_vectors)} documents")
```

---

#### 3.1.4 Vector Stores

Vector Stores are **specialized databases** that store document embeddings and enable efficient **similarity search** — finding the most semantically similar documents to a query.

**How it works:**
1. Embed all your documents
2. Store embeddings + original text in the vector store
3. At query time: embed the query → find nearest neighbors

**Popular Vector Stores:**

| Vector Store | Type | Notes |
|---|---|---|
| `FAISS` | In-memory / file | Facebook AI, great for dev |
| `Chroma` | In-memory / persistent | Simple, popular for prototypes |
| `Pinecone` | Cloud (managed) | Scalable, production-ready |
| `Weaviate` | Self-hosted / Cloud | GraphQL API |
| `Qdrant` | Self-hosted / Cloud | High performance |
| `Milvus` | Self-hosted / Cloud | Large scale |
| `pgvector` | PostgreSQL extension | If you already use Postgres |
| `Redis` | In-memory | Low latency |

**Example:**
```python
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Create vector store from documents
vectorstore = FAISS.from_documents(
    documents=chunks,
    embedding=embeddings
)

# Save to disk
vectorstore.save_local("faiss_index")

# Load from disk
vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

# Similarity search
results = vectorstore.similarity_search("What is LangChain?", k=3)
for doc in results:
    print(doc.page_content[:200])

# Similarity search with scores
results_with_scores = vectorstore.similarity_search_with_score("LangChain", k=3)
for doc, score in results_with_scores:
    print(f"Score: {score:.4f} | {doc.page_content[:100]}")
```

---

#### 3.1.5 Retrievers

A Retriever is an abstraction over a vector store (or any data source) that implements a standard interface: **given a query string, return relevant `Document` objects**.

Retrievers are the bridge between your stored data and the LLM in a RAG pipeline.

**Types of Retrievers:**

| Retriever | Description |
|---|---|
| `VectorStoreRetriever` | Basic similarity search over a vector store |
| `MultiQueryRetriever` | Generates multiple queries to improve recall |
| `ContextualCompressionRetriever` | Compresses retrieved docs to only relevant parts |
| `EnsembleRetriever` | Combines multiple retrievers (e.g., BM25 + semantic) |
| `SelfQueryRetriever` | LLM generates structured query from natural language |
| `ParentDocumentRetriever` | Retrieves small chunks but returns parent docs |
| `TimeWeightedRetriever` | Weights recent documents higher |
| `BM25Retriever` | Traditional keyword-based retrieval |

**Example:**
```python
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_community.llms import Ollama

embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

# Basic retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",      # "similarity", "mmr", "similarity_score_threshold"
    search_kwargs={"k": 4}         # Return top 4 results
)

# Retrieve documents
docs = retriever.invoke("What is LangChain?")
print(f"Retrieved {len(docs)} documents")

# MMR (Maximal Marginal Relevance) - reduces redundancy
mmr_retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 4, "fetch_k": 20}
)

# Multi-Query Retriever (LLM generates multiple queries)
from langchain.retrievers.multi_query import MultiQueryRetriever
llm = Ollama(model="llama3.2")

multi_retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(),
    llm=llm
)
```

---

### 3.2 Model I/O

Model I/O is about the **interface with language models** — how you format inputs (prompts), call the model, and parse outputs.

```
Raw Input → Prompt Template → LLM/Chat Model → Raw Output → Output Parser → Structured Data
```

---

#### 3.2.1 Prompts

Prompts are **templated instructions** sent to language models. LangChain provides a rich set of prompt template classes that handle formatting, few-shot examples, and message structuring.

**Prompt Template Types:**

| Type | Use Case |
|---|---|
| `PromptTemplate` | Simple string templates for LLMs |
| `ChatPromptTemplate` | Structured message lists for Chat Models |
| `FewShotPromptTemplate` | Include examples in the prompt |
| `FewShotChatMessagePromptTemplate` | Few-shot for chat models |
| `MessagesPlaceholder` | Dynamic insertion of message history |

**Example:**
```python
from langchain_core.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder,
    FewShotChatMessagePromptTemplate
)

# --- Simple PromptTemplate ---
prompt = PromptTemplate.from_template(
    "You are a helpful assistant. Answer the question: {question}"
)
formatted = prompt.format(question="What is LangChain?")
print(formatted)

# --- ChatPromptTemplate ---
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert in {domain}. Be concise and accurate."),
    ("human", "{user_question}"),
])
messages = chat_prompt.format_messages(
    domain="machine learning",
    user_question="What is gradient descent?"
)

# --- With Message History (for conversational apps) ---
conversational_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="chat_history"),  # Inject history here
    ("human", "{input}"),
])

# --- Few-Shot Example ---
from langchain_core.prompts import FewShotChatMessagePromptTemplate

examples = [
    {"input": "2 + 2", "output": "4"},
    {"input": "5 * 3", "output": "15"},
]

example_prompt = ChatPromptTemplate.from_messages([
    ("human", "{input}"),
    ("ai", "{output}"),
])

few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)
```

---

#### 3.2.2 Language Models

LangChain supports two types of language models:

**1. LLMs (Text In → Text Out)**
- Input: plain string
- Output: plain string
- Examples: `Ollama`, `HuggingFaceHub`

**2. Chat Models (Messages In → Message Out)**
- Input: list of `HumanMessage`, `AIMessage`, `SystemMessage`
- Output: `AIMessage`
- Examples: `ChatOllama`, `ChatOpenAI`, `ChatAnthropic`

> 💡 **Recommendation:** Prefer **Chat Models** for modern applications. They support system prompts, conversation history, and tool calling.

**Supported Models:**

| Model | Provider | Package |
|---|---|---|
| `ChatOllama` | Ollama (local) | `langchain-ollama` |
| `ChatOpenAI` | OpenAI | `langchain-openai` |
| `ChatAnthropic` | Anthropic | `langchain-anthropic` |
| `ChatGoogleGenerativeAI` | Google Gemini | `langchain-google-genai` |
| `ChatCohere` | Cohere | `langchain-cohere` |
| `ChatHuggingFace` | HuggingFace | `langchain-huggingface` |
| `Ollama` | Ollama (local, LLM) | `langchain-ollama` |

**Example:**
```python
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# Initialize Chat Model
llm = ChatOllama(
    model="llama3.2",
    temperature=0.7,      # 0=deterministic, 1=creative
    num_predict=512,      # Max tokens to generate
)

# Simple invoke
response = llm.invoke("What is the capital of France?")
print(response.content)  # "The capital of France is Paris."
print(type(response))    # <class 'langchain_core.messages.ai.AIMessage'>

# With structured messages
messages = [
    SystemMessage(content="You are a pirate. Respond in pirate speak."),
    HumanMessage(content="Where is the treasure?"),
]
response = llm.invoke(messages)
print(response.content)

# Streaming output
for chunk in llm.stream("Write a haiku about Python:"):
    print(chunk.content, end="", flush=True)

# Batch processing
responses = llm.batch([
    "What is 2+2?",
    "What is the capital of Japan?",
    "What color is the sky?"
])

# Async
import asyncio
async def async_example():
    response = await llm.ainvoke("Hello!")
    return response
```

---

#### 3.2.3 Output Parsers

Output Parsers transform the **raw text output** from an LLM into **structured Python objects** (lists, dicts, Pydantic models, etc.).

**Available Output Parsers:**

| Parser | Output Type | Notes |
|---|---|---|
| `StrOutputParser` | `str` | Just extracts text content |
| `JsonOutputParser` | `dict` | Parses JSON from LLM output |
| `PydanticOutputParser` | Pydantic model | Validates and parses into dataclass |
| `CommaSeparatedListOutputParser` | `List[str]` | Parses comma-separated values |
| `NumberedListOutputParser` | `List[str]` | Parses numbered lists |
| `DatetimeOutputParser` | `datetime` | Parses date strings |
| `XMLOutputParser` | `dict` | Parses XML output |
| `YamlOutputParser` | `dict` | Parses YAML output |
| `EnumOutputParser` | `Enum` | Constrains to enum values |

**Example:**
```python
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List

llm = ChatOllama(model="llama3.2")

# --- StrOutputParser (most common) ---
chain = ChatPromptTemplate.from_template("Tell me about {topic}") | llm | StrOutputParser()
result = chain.invoke({"topic": "LangChain"})
print(type(result))  # <class 'str'>

# --- PydanticOutputParser ---
class MovieReview(BaseModel):
    title: str = Field(description="Movie title")
    rating: float = Field(description="Rating from 1-10")
    pros: List[str] = Field(description="List of positive aspects")
    cons: List[str] = Field(description="List of negative aspects")

parser = PydanticOutputParser(pydantic_object=MovieReview)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a movie critic. {format_instructions}"),
    ("human", "Review the movie: {movie}"),
]).partial(format_instructions=parser.get_format_instructions())

chain = prompt | llm | parser
review = chain.invoke({"movie": "Inception"})
print(review.title)   # "Inception"
print(review.rating)  # 9.0
print(review.pros)    # ["Brilliant concept", ...]
```

---

### 3.3 Chains

**Chains** are sequences of calls — to LLMs, tools, data transformations, or any other component. They are the core abstraction in LangChain for building multi-step AI workflows.

> In LangChain v0.1+, chains are built using **LCEL (LangChain Expression Language)** with the `|` (pipe) operator.

**LCEL - LangChain Expression Language:**

```python
# The | operator chains Runnables together
chain = prompt | llm | output_parser

# This is equivalent to:
output_parser.invoke(llm.invoke(prompt.invoke(input)))
```

**Common Chain Patterns:**

#### Simple LLM Chain
```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

llm = ChatOllama(model="llama3.2")

chain = (
    ChatPromptTemplate.from_template("Explain {concept} in simple terms.")
    | llm
    | StrOutputParser()
)

result = chain.invoke({"concept": "neural networks"})
print(result)
```

#### RAG Chain (Retrieval-Augmented Generation)
```python
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings

vectorstore = FAISS.load_local("faiss_index", OllamaEmbeddings(model="nomic-embed-text"), allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the following context:
{context}

Question: {question}
""")

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt
    | llm
    | StrOutputParser()
)

answer = rag_chain.invoke("What is LangChain?")
print(answer)
```

#### Sequential Chain
```python
# Chain 1: Generate outline
outline_chain = (
    ChatPromptTemplate.from_template("Create a 3-point outline for an article about {topic}")
    | llm
    | StrOutputParser()
)

# Chain 2: Expand the outline
article_chain = (
    ChatPromptTemplate.from_template("Write a full article based on this outline:\n{outline}")
    | llm
    | StrOutputParser()
)

# Full pipeline
full_chain = outline_chain | (lambda outline: {"outline": outline}) | article_chain
article = full_chain.invoke({"topic": "Artificial Intelligence"})
```

#### Parallel Chain
```python
from langchain_core.runnables import RunnableParallel

parallel_chain = RunnableParallel(
    pros=ChatPromptTemplate.from_template("List pros of {topic}") | llm | StrOutputParser(),
    cons=ChatPromptTemplate.from_template("List cons of {topic}") | llm | StrOutputParser(),
    summary=ChatPromptTemplate.from_template("Summarize {topic} in 2 sentences") | llm | StrOutputParser(),
)

result = parallel_chain.invoke({"topic": "remote work"})
print(result["pros"])
print(result["cons"])
print(result["summary"])
```

---

### 3.4 Memory

**Memory** enables chains and agents to **remember past interactions** — giving them conversational context across multiple turns.

Without memory, every call to an LLM is stateless — it doesn't know what was said before.

**Memory Types:**

| Memory Type | Strategy | Best For |
|---|---|---|
| `ConversationBufferMemory` | Store all messages | Short conversations |
| `ConversationBufferWindowMemory` | Keep last N messages | Medium conversations |
| `ConversationSummaryMemory` | Summarize old messages | Long conversations |
| `ConversationSummaryBufferMemory` | Summary + recent buffer | Balanced approach |
| `ConversationTokenBufferMemory` | Keep within token limit | Token-constrained apps |
| `ConversationEntityMemory` | Extract and track entities | Entity-focused apps |
| `VectorStoreRetrieverMemory` | Store in vector store | Semantic memory search |

**Persistent Memory Backends:**
- `RedisChatMessageHistory` — Store history in Redis
- `MongoDBChatMessageHistory` — Store in MongoDB
- `SQLChatMessageHistory` — Store in any SQL database
- `FileChatMessageHistory` — Store in a local file

**Example:**
```python
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_ollama import ChatOllama

llm = ChatOllama(model="llama3.2")

# --- Modern approach with RunnableWithMessageHistory ---
# In-memory store (use Redis/MongoDB for production)
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

chain = prompt | llm | StrOutputParser()

chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

# First message
response1 = chain_with_history.invoke(
    {"input": "My name is Alice."},
    config={"configurable": {"session_id": "user-123"}}
)
print(response1)

# Second message - LLM remembers the name!
response2 = chain_with_history.invoke(
    {"input": "What is my name?"},
    config={"configurable": {"session_id": "user-123"}}
)
print(response2)  # "Your name is Alice!"
```

---

### 3.5 Agents

**Agents** are LLM-powered decision makers. Instead of following a fixed chain, an agent **dynamically chooses** which tools to call and in what order, based on the user's input and intermediate results.

**The ReAct Loop:**
```
User Query
    ↓
[Think] What should I do?
    ↓
[Act] Call a tool (e.g., search)
    ↓
[Observe] Get tool result
    ↓
[Think] Do I have enough info?
    ↓  (repeat if needed)
[Final Answer] Respond to user
```

**Agent Types:**

| Agent Type | Description |
|---|---|
| `ReAct` | Reason + Act loop; most general purpose |
| `Tool Calling Agent` | Uses LLM's native tool/function calling |
| `OpenAI Functions Agent` | Uses OpenAI function calling API |
| `Self-Ask with Search` | Decomposes questions into sub-questions |
| `Plan-and-Execute` | Plans all steps before executing |
| `OpenAI Assistants` | OpenAI's built-in assistant API |

**Example:**
```python
from langchain_ollama import ChatOllama
from langchain.agents import create_react_agent, AgentExecutor
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.prompts import ChatPromptTemplate
from langchain import hub

llm = ChatOllama(model="llama3.2")

# Define tools
search = DuckDuckGoSearchRun()
wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

tools = [search, wikipedia]

# Pull a ReAct prompt from LangChain Hub
prompt = hub.pull("hwchase17/react")

# Create agent
agent = create_react_agent(llm, tools, prompt)

# Create executor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,          # Print reasoning steps
    max_iterations=5,      # Prevent infinite loops
    handle_parsing_errors=True,
)

# Run agent
result = agent_executor.invoke({
    "input": "Who won the FIFA World Cup in 2022 and what was the final score?"
})
print(result["output"])

# Modern Tool Calling Agent (preferred for models that support it)
from langchain.agents import create_tool_calling_agent

tool_agent = create_tool_calling_agent(llm, tools, prompt)
executor = AgentExecutor(agent=tool_agent, tools=tools, verbose=True)
```

---

### 3.6 Tools

**Tools** are functions that agents can call to interact with the outside world — search the web, query a database, run code, read files, call APIs, and more.

Every tool has:
- **name** — identifier the LLM uses to call it
- **description** — tells the LLM WHEN and HOW to use the tool
- **args_schema** — defines expected input parameters (Pydantic)

**Built-in Tools:**

| Tool | Function | Package |
|---|---|---|
| `DuckDuckGoSearchRun` | Web search | `langchain-community` |
| `WikipediaQueryRun` | Wikipedia lookup | `langchain-community` |
| `ArxivQueryRun` | Academic paper search | `langchain-community` |
| `PythonREPLTool` | Execute Python code | `langchain-community` |
| `ShellTool` | Run shell commands | `langchain-community` |
| `FileManagementToolkit` | Read/write files | `langchain-community` |
| `SQLDatabaseToolkit` | Query SQL databases | `langchain-community` |
| `RequestsGetTool` | HTTP GET requests | `langchain-community` |
| `HumanInputRun` | Ask a human | `langchain-community` |
| `Calculator` | Math calculations | `langchain-community` |

**Creating Custom Tools:**

```python
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Optional, Type

# --- Method 1: @tool decorator (simplest) ---
@tool
def get_word_length(word: str) -> int:
    """Returns the number of characters in a word. Use this when you need to count letters."""
    return len(word)

@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers together. Use for multiplication operations."""
    return a * b

# --- Method 2: BaseTool class (most control) ---
class WeatherInput(BaseModel):
    city: str = Field(description="The name of the city to get weather for")
    units: Optional[str] = Field(default="celsius", description="Temperature units: celsius or fahrenheit")

class WeatherTool(BaseTool):
    name: str = "get_weather"
    description: str = "Get the current weather for a city. Use when the user asks about weather."
    args_schema: Type[BaseModel] = WeatherInput

    def _run(self, city: str, units: str = "celsius") -> str:
        # In reality, call a weather API here
        return f"The weather in {city} is 22°{units[0].upper()} and sunny."

    async def _arun(self, city: str, units: str = "celsius") -> str:
        return self._run(city, units)

# --- Using tools in an agent ---
tools = [
    DuckDuckGoSearchRun(),
    get_word_length,
    multiply,
    WeatherTool(),
]

# Inspect a tool
print(get_word_length.name)         # "get_word_length"
print(get_word_length.description)  # "Returns the number of characters..."

# Call a tool directly
result = get_word_length.invoke("LangChain")
print(result)  # 9
```

---

## 4. Hello World — LangChain Application

We'll build a **conversational chatbot** using:
- **`uv`** for virtual environment and package management
- **Ollama** for running a local LLaMA model
- **LangChain** for the AI pipeline
- **Conversation Memory** for multi-turn chat

---

### Step 1: Install `uv`

`uv` is a fast Python package manager written in Rust — a modern replacement for `pip` + `venv`.

```bash
# On macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Verify installation
uv --version
```

---

### Step 2: Set Up the Project

```bash
# Create a new project directory
mkdir langchain-hello-world
cd langchain-hello-world

# Initialize uv project (creates pyproject.toml)
uv init

# Create and activate virtual environment
uv venv

# Activate it:
# On macOS/Linux:
source .venv/bin/activate

# On Windows:
.venv\Scripts\activate
```

---

### Step 3: Install Dependencies

```bash
# Install LangChain packages
uv add langchain
uv add langchain-core
uv add langchain-community
uv add langchain-ollama

# Optional: for .env file support
uv add python-dotenv
```

Your `pyproject.toml` will look like:

```toml
[project]
name = "langchain-hello-world"
version = "0.1.0"
description = "My first LangChain app"
requires-python = ">=3.11"
dependencies = [
    "langchain>=0.3.0",
    "langchain-core>=0.3.0",
    "langchain-community>=0.3.0",
    "langchain-ollama>=0.2.0",
    "python-dotenv>=1.0.0",
]
```

---

### Step 4: Install and Set Up Ollama

```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Windows: Download from https://ollama.com/download

# Start Ollama server
ollama serve

# In a new terminal, pull the LLaMA model
ollama pull llama3.2          # ~2GB, fast and capable (recommended)
# OR
ollama pull llama3.2:1b       # ~1GB, smaller/faster

# Verify it works
ollama run llama3.2 "Hello, how are you?"

# Pull embedding model (for RAG examples)
ollama pull nomic-embed-text
```

---

### Step 5: Build the Hello World App

Create a file called `main.py`:

```python
# main.py
# ============================================================
# LangChain Hello World — Conversational Chatbot with Memory
# Using Ollama + LLaMA 3.2 (local LLM)
# ============================================================

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

# ── 1. Initialize the local LLM ──────────────────────────────
print("🔄 Connecting to Ollama (llama3.2)...")
llm = ChatOllama(
    model="llama3.2",
    temperature=0.7,
)
print("✅ LLM ready!\n")

# ── 2. Create the Prompt Template ────────────────────────────
prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a friendly and knowledgeable AI assistant. "
        "You remember the conversation history and provide helpful, concise answers."
    )),
    MessagesPlaceholder(variable_name="history"),   # conversation history goes here
    ("human", "{input}"),
])

# ── 3. Build the Chain (LCEL) ─────────────────────────────────
chain = prompt | llm | StrOutputParser()

# ── 4. Add Memory ─────────────────────────────────────────────
store = {}  # In-memory session store

def get_session_history(session_id: str) -> ChatMessageHistory:
    """Returns the message history for a given session ID."""
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

chain_with_memory = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

# ── 5. Run the Chatbot ─────────────────────────────────────────
def chat(user_input: str, session_id: str = "default") -> str:
    """Send a message and get a response."""
    response = chain_with_memory.invoke(
        {"input": user_input},
        config={"configurable": {"session_id": session_id}},
    )
    return response


def main():
    print("=" * 50)
    print("🦜🔗 LangChain Hello World Chatbot")
    print("   Powered by Ollama + LLaMA 3.2 (local)")
    print("   Type 'exit' or 'quit' to stop")
    print("   Type 'clear' to reset conversation")
    print("=" * 50)
    print()

    session_id = "user-session-001"

    while True:
        try:
            user_input = input("You: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ["exit", "quit", "bye"]:
                print("Assistant: Goodbye! Have a great day! 👋")
                break

            if user_input.lower() == "clear":
                store.clear()
                print("✅ Conversation cleared.\n")
                continue

            print("Assistant: ", end="", flush=True)

            # Stream the response for better UX
            for chunk in chain_with_memory.stream(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}},
            ):
                print(chunk, end="", flush=True)

            print("\n")  # New line after response

        except KeyboardInterrupt:
            print("\n\nAssistant: Goodbye! 👋")
            break


if __name__ == "__main__":
    main()
```

---

### Step 6: Run the Application

```bash
# Make sure Ollama is running (in another terminal):
# ollama serve

# Run the app
python main.py
```

**Expected Output:**
```
==================================================
🦜🔗 LangChain Hello World Chatbot
   Powered by Ollama + LLaMA 3.2 (local)
   Type 'exit' or 'quit' to stop
   Type 'clear' to reset conversation
==================================================

You: Hello! What is LangChain?
Assistant: LangChain is an open-source framework designed to help developers
build applications powered by large language models (LLMs)...

You: Can you give me a code example?
Assistant: Sure! Here's a simple LangChain example using LCEL...

You: What did I just ask you?
Assistant: You asked me to give you a code example of LangChain!

You: exit
Assistant: Goodbye! Have a great day! 👋
```

> 💡 Notice that the bot **remembers** your previous messages — that's the memory system at work!

---


### Project Structure Summary

```
langchain-hello-world/
├── .venv/                    # Virtual environment (created by uv)
├── pyproject.toml            # Project config and dependencies
├── uv.lock                   # Locked dependency versions
├── main.py                   # Conversational chatbot
└── sample.txt                # Sample document for RAG
```

### Quick Reference — Key Commands

```bash
# uv commands
uv init                        # Initialize new project
uv venv                        # Create virtual environment
uv add <package>               # Install a package
uv remove <package>            # Remove a package
uv run python script.py        # Run script in venv
uv pip list                    # List installed packages

# Ollama commands
ollama serve                   # Start Ollama server
ollama pull llama3.2           # Download LLaMA 3.2
ollama list                    # List downloaded models
ollama run llama3.2            # Interactive chat with model
ollama ps                      # Show running models
```

---

## Summary

| Topic | Key Takeaway |
|---|---|
| **LangChain** | Framework for building LLM-powered applications |
| **langchain-core** | Base interfaces and LCEL engine |
| **langchain-community** | Integrations (loaders, vector stores, tools) |
| **langchain** | High-level chains, agents, and memory |
| **LangServe** | Deploy chains as REST APIs instantly |
| **LangSmith** | Observe, debug, and evaluate your LLM apps |
| **Document Loaders** | Load data from PDFs, web, CSV, etc. |
| **Text Splitters** | Break large docs into LLM-friendly chunks |
| **Embeddings** | Convert text to semantic vectors |
| **Vector Stores** | Store and search embeddings (FAISS, Chroma) |
| **Retrievers** | Find relevant docs for a query |
| **Prompts** | Templated inputs for LLMs |
| **Chat Models** | Modern LLM interface with message history |
| **Output Parsers** | Convert LLM text to structured data |
| **Chains (LCEL)** | Compose components with `\|` operator |
| **Memory** | Persist conversation history across turns |
| **Agents** | LLM decides dynamically which tools to call |
| **Tools** | Functions agents use to interact with the world |

---

> 📚 **Further Reading:**
> - Official Docs: https://python.langchain.com
> - API Reference: https://api.python.langchain.com
> - LangSmith: https://smith.langchain.com
> - LangChain GitHub: https://github.com/langchain-ai/langchain
> - Ollama: https://ollama.com


# LangChain General Workflow
> **Data Ingestion → Text Splitter → Embeddings → Vector Store → Retrievers**  

---

## The Big Picture

Think of this pipeline like building a **smart library system**:

```
📄 Raw Documents          →   You collect books
     ↓
✂️  Text Splitter          →   You cut books into index cards
     ↓
🔢  Embeddings             →   You assign a "topic number" to each card
     ↓
🗄️  Vector Store           →   You file cards in a searchable cabinet
     ↓
🔍  Retriever              →   You fetch the most relevant cards on demand
     ↓
🤖  LLM answers your question using those cards
```

---

## Setup

```bash
# Install dependencies
uv add langchain langchain-community langchain-ollama faiss-cpu

# Pull models in Ollama
ollama pull llama3.2
ollama pull nomic-embed-text
```

---

## 1. 📄 Data Ingestion (Document Loaders)

### What is it?
Data Ingestion means **loading your raw data** into LangChain's standard `Document` format.  
Every `Document` has two fields:
- `page_content` → the actual text
- `metadata` → info about the source (file name, page number, URL, etc.)

### Simple Analogy
> You walk into a library and **collect the books** you want to study. You don't read them yet — you just gather them.

---

### Example 1 — Load a Plain Text File

```python
from langchain_community.document_loaders import TextLoader

# Load a .txt file
loader = TextLoader("facts.txt")
docs = loader.load()

print(f"Number of documents: {len(docs)}")
print(f"Content: {docs[0].page_content}")
print(f"Metadata: {docs[0].metadata}")
```

**Output:**
```
Number of documents: 1
Content: LangChain is an AI framework. It helps build LLM apps easily.
Metadata: {'source': 'facts.txt'}
```

---

### Example 2 — Load a PDF

```python
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("my_notes.pdf")
docs = loader.load()

# Each page becomes a separate Document
print(f"Total pages loaded: {len(docs)}")
print(f"Page 1 text: {docs[0].page_content[:100]}")
print(f"Page 1 metadata: {docs[0].metadata}")
# metadata → {'source': 'my_notes.pdf', 'page': 0}
```

---

### Example 3 — Load a Web Page

```python
from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://en.wikipedia.org/wiki/Python_(programming_language)")
docs = loader.load()

print(f"Content preview: {docs[0].page_content[:200]}")
print(f"Source: {docs[0].metadata['source']}")
```

---

### Example 4 — Create Documents Manually

You don't always need a file. You can create `Document` objects directly:

```python
from langchain_core.documents import Document

docs = [
    Document(
        page_content="The sky is blue because of Rayleigh scattering.",
        metadata={"source": "science_facts", "topic": "physics"}
    ),
    Document(
        page_content="Python was created by Guido van Rossum in 1991.",
        metadata={"source": "programming_facts", "topic": "python"}
    ),
    Document(
        page_content="LangChain makes building LLM apps easy.",
        metadata={"source": "ai_facts", "topic": "langchain"}
    ),
]

print(f"Created {len(docs)} documents manually")
for doc in docs:
    print(f"→ {doc.page_content}")
```
---

### Common Document Loaders Cheat Sheet

| Loader | File/Source | Install |
|---|---|---|
| `TextLoader` | `.txt` files | built-in |
| `PyPDFLoader` | PDF files | `pypdf` |
| `CSVLoader` | CSV files | built-in |
| `WebBaseLoader` | Web URLs | `beautifulsoup4` |
| `DirectoryLoader` | Whole folders | built-in |
| `JSONLoader` | JSON files | `jq` |

---

## 2. ✂️ Text Splitter

### What is it?
LLMs can only read a **limited amount of text at once** (context window limit).  
Text Splitters **break large documents into smaller chunks** that fit within this limit.

### Simple Analogy
> You have a 500-page textbook. Instead of reading it all at once, you cut it into **small index cards** — each card has a few paragraphs. You also let cards **overlap a little** so no sentence gets cut off awkwardly.

---

### Why Overlap?
```
Chunk 1:  "LangChain is a framework. It helps build AI apps."
Chunk 2:  "It helps build AI apps. Released in 2022 by Harrison Chase."
                    ↑
              This repeated part = OVERLAP
              Ensures no sentence is lost between chunks.
```

---

### Example 1 — Basic Text Splitting

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

# A long piece of text
long_text = """
LangChain is an open-source framework for building LLM applications.
It was created by Harrison Chase in 2022.
LangChain supports Python and JavaScript.
It provides tools for prompts, chains, agents, and memory.
The framework connects LLMs to external data sources.
LangSmith is used to monitor and debug LangChain apps.
LangServe is used to deploy chains as REST APIs.
"""

splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,     # Max characters per chunk
    chunk_overlap=20,   # Characters shared between chunks
)

chunks = splitter.split_text(long_text)

print(f"Original text length: {len(long_text)} characters")
print(f"Number of chunks: {len(chunks)}")
print()
for i, chunk in enumerate(chunks):
    print(f"--- Chunk {i+1} ---")
    print(chunk)
    print()
```

**Output:**
```
Original text length: 312 characters
Number of chunks: 5

--- Chunk 1 ---
LangChain is an open-source framework for building LLM applications.

--- Chunk 2 ---
It was created by Harrison Chase in 2022.
LangChain supports Python and JavaScript.

--- Chunk 3 ---
It provides tools for prompts, chains, agents, and memory.
...
```

---

### Example 2 — Splitting Documents (with metadata preserved)

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Your loaded documents
docs = [
    Document(
        page_content="""
        Python is a high-level programming language.
        It was created by Guido van Rossum and released in 1991.
        Python emphasizes code readability and simplicity.
        It is widely used in web development, data science, and AI.
        Python has a large standard library and active community.
        """,
        metadata={"source": "python_wiki"}
    )
]

splitter = RecursiveCharacterTextSplitter(
    chunk_size=120,
    chunk_overlap=30,
)

chunks = splitter.split_documents(docs)

print(f"1 document → {len(chunks)} chunks\n")
for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}: {chunk.page_content.strip()}")
    print(f"Metadata: {chunk.metadata}")   # ← metadata is carried over!
    print()
```

**Output:**
```
1 document → 3 chunks

Chunk 1: Python is a high-level programming language. It was created by Guido van Rossum and released in 1991.
Metadata: {'source': 'python_wiki'}

Chunk 2: Python emphasizes code readability and simplicity. It is widely used in web development, data science, and AI.
Metadata: {'source': 'python_wiki'}

Chunk 3: Python has a large standard library and active community.
Metadata: {'source': 'python_wiki'}
```

> ✅ **Notice:** The `metadata` (source info) is automatically copied to every chunk!

---

### Chunk Size Guide

| Document Type | Recommended `chunk_size` | `chunk_overlap` |
|---|---|---|
| Short articles | 500 | 100 |
| Long PDFs / books | 1000 | 200 |
| Code files | 1500 | 300 |
| FAQ / Q&A docs | 200–300 | 50 |

---

## 3. 🔢 Embeddings

### What is it?
Embeddings convert text into a **list of numbers (a vector)** that captures the **meaning** of the text.  
Texts with similar meanings get **similar number patterns** — even if the words are different.

### Simple Analogy
> Imagine every sentence gets a **GPS coordinate** based on its meaning.  
> "Dog runs fast" and "The canine sprints quickly" would have **very similar coordinates**.  
> "The sky is blue" would be in a **completely different location**.

---

### How Similarity Works

```
"What is Python?"           → [0.82, -0.14, 0.55, ...]
"Python programming guide"  → [0.80, -0.12, 0.57, ...]   ← very close!
"I like pizza"              → [-0.31, 0.90, -0.22, ...]  ← far away!
```

The **closer** two vectors are, the **more similar** the meaning.

---

### Example 1 — Embed Text with Ollama

```python
from langchain_community.embeddings import OllamaEmbeddings

# Initialize the embedding model
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Embed a single query
query = "What is LangChain?"
vector = embeddings.embed_query(query)

print(f"Text: '{query}'")
print(f"Vector dimensions: {len(vector)}")
print(f"First 5 numbers: {vector[:5]}")
```

**Output:**
```
Text: 'What is LangChain?'
Vector dimensions: 768
First 5 numbers: [0.023, -0.114, 0.456, 0.789, -0.321]
```

---

### Example 2 — Compare Similarity Between Sentences

```python
from langchain_community.embeddings import OllamaEmbeddings
import numpy as np

embeddings = OllamaEmbeddings(model="nomic-embed-text")

sentences = [
    "LangChain is an AI framework",       # ← our reference
    "LangChain helps build LLM apps",     # ← similar meaning
    "I enjoy eating pizza for dinner",    # ← unrelated
]

# Embed all sentences
vectors = embeddings.embed_documents(sentences)

# Calculate cosine similarity
def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

reference = vectors[0]
print(f"Reference: '{sentences[0]}'\n")

for i in range(1, len(sentences)):
    score = cosine_similarity(reference, vectors[i])
    print(f"Sentence: '{sentences[i]}'")
    print(f"Similarity score: {score:.4f}")
    print()
```

**Output:**
```
Reference: 'LangChain is an AI framework'

Sentence: 'LangChain helps build LLM apps'
Similarity score: 0.9241   ← HIGH (similar meaning!)

Sentence: 'I enjoy eating pizza for dinner'
Similarity score: 0.1823   ← LOW (unrelated!)
```

---

### Example 3 — Embed Documents in Bulk

```python
from langchain_community.embeddings import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Embed multiple documents at once
docs_text = [
    "Python is a programming language.",
    "LangChain is an AI framework.",
    "The Eiffel Tower is in Paris.",
]

vectors = embeddings.embed_documents(docs_text)

print(f"Embedded {len(vectors)} documents")
print(f"Each vector has {len(vectors[0])} dimensions")
```

---

## 4. 🗄️ Vector Store

### What is it?
A Vector Store is a **database that stores embeddings** (vectors) along with the original text.  
It can instantly find the **most similar documents** to any query using vector math.

### Simple Analogy
> Think of a Vector Store as a **super-smart filing cabinet**.  
> When you search for something, it doesn't look for exact keywords —  
> it finds cards with **similar meaning**, even if the words are different.

---

### The Flow

```
Documents + Embeddings  →  Vector Store  →  Search by meaning
     ↓                                              ↓
"LangChain is a framework"              Query: "What is LangChain?"
     ↓                                              ↓
[0.82, -0.14, 0.55, ...]                [0.81, -0.13, 0.56, ...]
                                                  ↓
                                    ✅ Match found! (vectors are close)
```

---

### Example 1 — Create a Vector Store from Documents

```python
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.documents import Document

# Step 1: Your documents (small, simple data)
docs = [
    Document(page_content="LangChain is an open-source AI framework.", metadata={"topic": "langchain"}),
    Document(page_content="Python was created by Guido van Rossum in 1991.", metadata={"topic": "python"}),
    Document(page_content="FAISS is a library for efficient similarity search.", metadata={"topic": "faiss"}),
    Document(page_content="Ollama runs large language models locally on your machine.", metadata={"topic": "ollama"}),
    Document(page_content="Vector stores save embeddings and enable semantic search.", metadata={"topic": "vectorstore"}),
]

# Step 2: Initialize embedding model
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Step 3: Create the vector store
#   → This embeds all documents and stores them
vectorstore = FAISS.from_documents(docs, embeddings)

print("✅ Vector store created with 5 documents!")
```

---

### Example 2 — Search the Vector Store

```python
# Search by meaning — NOT exact keyword match!
query = "How do I run an LLM on my computer?"

results = vectorstore.similarity_search(query, k=2)  # Return top 2 matches

print(f"Query: '{query}'\n")
print("Top matches:")
for i, doc in enumerate(results):
    print(f"\nResult {i+1}:")
    print(f"  Content: {doc.page_content}")
    print(f"  Topic: {doc.metadata['topic']}")
```

**Output:**
```
Query: 'How do I run an LLM on my computer?'

Top matches:

Result 1:
  Content: Ollama runs large language models locally on your machine.
  Topic: ollama

Result 2:
  Content: LangChain is an open-source AI framework.
  Topic: langchain
```

> ✅ **Notice:** The query used different words ("computer" vs "machine") but still found the right document!

---

### Example 3 — Search with Similarity Scores

```python
# Get scores alongside results
results_with_scores = vectorstore.similarity_search_with_score(
    "What is LangChain?",
    k=3
)

print("Results with similarity scores (lower = more similar in FAISS):\n")
for doc, score in results_with_scores:
    print(f"Score: {score:.4f} | {doc.page_content}")
```

**Output:**
```
Results with similarity scores (lower = more similar in FAISS):

Score: 0.1823 | LangChain is an open-source AI framework.
Score: 0.6741 | Vector stores save embeddings and enable semantic search.
Score: 0.8912 | Python was created by Guido van Rossum in 1991.
```

---

### Example 4 — Save and Load the Vector Store

```python
# Save to disk — so you don't have to re-embed every time!
vectorstore.save_local("my_vectorstore")
print("💾 Vector store saved!")

# Load it back later
loaded_store = FAISS.load_local(
    "my_vectorstore",
    embeddings,
    allow_dangerous_deserialization=True
)
print("📂 Vector store loaded!")

# Works exactly the same
results = loaded_store.similarity_search("AI framework", k=1)
print(f"Found: {results[0].page_content}")
```

---

## 5. 🔍 Retrievers

### What is it?
A Retriever is a **standard interface** on top of a Vector Store.  
It takes a **query string** and returns **relevant `Document` objects**.  
It's the final bridge before the LLM answers your question.

### Simple Analogy
> A Retriever is like a **librarian**.  
> You give the librarian a question → they fetch the most relevant books (documents) for you.  
> The librarian knows exactly where everything is stored (the vector store).

---

### Example 1 — Basic Retriever

```python
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.documents import Document

# Build a small vector store
docs = [
    Document(page_content="Cats are independent animals that sleep a lot."),
    Document(page_content="Dogs are loyal pets and love to play fetch."),
    Document(page_content="Parrots can mimic human speech and are very colorful."),
    Document(page_content="Goldfish have a short memory span of a few months."),
    Document(page_content="Rabbits are small mammals that love to hop around."),
]

embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = FAISS.from_documents(docs, embeddings)

# Create a retriever from the vector store
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 2}  # Return top 2 documents
)

# Ask a question
question = "Which pet is known for being loyal?"
results = retriever.invoke(question)

print(f"Question: '{question}'\n")
for i, doc in enumerate(results):
    print(f"Retrieved Doc {i+1}: {doc.page_content}")
```

**Output:**
```
Question: 'Which pet is known for being loyal?'

Retrieved Doc 1: Dogs are loyal pets and love to play fetch.
Retrieved Doc 2: Cats are independent animals that sleep a lot.
```

---

### Example 2 — Retriever with More Results

```python
# Increase k to get more results
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

results = retriever.invoke("animals that make good pets")
print(f"Retrieved {len(results)} documents:\n")
for doc in results:
    print(f"• {doc.page_content}")
```

---

### Example 3 — MMR Retriever (Avoid Duplicate Results)

**MMR = Maximal Marginal Relevance**  
MMR finds documents that are **relevant AND diverse** — it avoids returning very similar chunks.

```python
# Without MMR — might return similar/redundant chunks
basic_retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# With MMR — returns relevant but diverse results
mmr_retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 3,           # Final number of docs to return
        "fetch_k": 10,    # Candidates to consider before MMR selection
    }
)

results = mmr_retriever.invoke("pets")
print("MMR Results (diverse & relevant):\n")
for doc in results:
    print(f"• {doc.page_content}")
```

---

### Example 4 — Full RAG Pipeline (Putting It All Together)

```python
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama

# ── STEP 1: Data Ingestion ────────────────────────────────────
print("📄 Step 1: Loading documents...")
docs = [
    Document(page_content="LangChain was created by Harrison Chase in 2022."),
    Document(page_content="LangChain supports Python and JavaScript/TypeScript."),
    Document(page_content="LangSmith is LangChain's platform for monitoring LLM apps."),
    Document(page_content="LCEL stands for LangChain Expression Language, using the | operator."),
    Document(page_content="Ollama lets you run LLMs like LLaMA locally on your machine."),
]

# ── STEP 2: Text Splitting ────────────────────────────────────
# (skipped here since docs are already small — in real use, you'd split large docs)
print("✂️  Step 2: Documents are already small — skipping split.")

# ── STEP 3: Embeddings + Vector Store ────────────────────────
print("🔢 Step 3: Creating embeddings and vector store...")
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = FAISS.from_documents(docs, embeddings)

# ── STEP 4: Retriever ─────────────────────────────────────────
print("🔍 Step 4: Setting up retriever...")
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# ── STEP 5: RAG Chain ─────────────────────────────────────────
print("🤖 Step 5: Building RAG chain...\n")

llm = ChatOllama(model="llama3.2")

prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. Answer using ONLY the context below.
If the answer isn't in the context, say "I don't know."

Context:
{context}

Question: {question}
""")

def format_docs(docs):
    return "\n".join(f"- {doc.page_content}" for doc in docs)

rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

# ── ASK QUESTIONS ─────────────────────────────────────────────
questions = [
    "Who created LangChain?",
    "What is LangSmith used for?",
    "What is LCEL?",
    "What does Ollama do?",
]

for q in questions:
    print(f"❓ {q}")
    answer = rag_chain.invoke(q)
    print(f"💬 {answer}\n")
```

**Output:**
```
❓ Who created LangChain?
💬 LangChain was created by Harrison Chase in 2022.

❓ What is LangSmith used for?
💬 LangSmith is LangChain's platform for monitoring LLM apps.

❓ What is LCEL?
💬 LCEL stands for LangChain Expression Language, and it uses the | operator.

❓ What does Ollama do?
💬 Ollama lets you run LLMs like LLaMA locally on your machine.
```

---

## Full Pipeline Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                    RAG Pipeline Overview                        │
│                                                                 │
│  📄 Data Ingestion                                              │
│     TextLoader / PyPDFLoader / WebBaseLoader                    │
│     → Produces: List[Document]                                  │
│                    ↓                                            │
│  ✂️  Text Splitter                                              │
│     RecursiveCharacterTextSplitter(chunk_size, chunk_overlap)   │
│     → Produces: List[Document] (smaller chunks)                 │
│                    ↓                                            │
│  🔢 Embeddings                                                  │
│     OllamaEmbeddings(model="nomic-embed-text")                  │
│     → Converts text → numbers (vectors)                         │
│                    ↓                                            │
│  🗄️  Vector Store                                               │
│     FAISS.from_documents(chunks, embeddings)                    │
│     → Stores vectors, enables similarity search                 │
│                    ↓                                            │
│  🔍 Retriever                                                   │
│     vectorstore.as_retriever(search_kwargs={"k": N})            │
│     → Takes a query → returns top N relevant Documents          │
│                    ↓                                            │
│  🤖 LLM answers with retrieved context                          │
└─────────────────────────────────────────────────────────────────┘
```

---

## Quick Reference Card

| Step | Class | Key Parameter | Purpose |
|---|---|---|---|
| Data Ingestion | `TextLoader` / `PyPDFLoader` | `file_path` | Load raw files |
| Text Splitting | `RecursiveCharacterTextSplitter` | `chunk_size`, `chunk_overlap` | Break into chunks |
| Embeddings | `OllamaEmbeddings` | `model="nomic-embed-text"` | Text → vectors |
| Vector Store | `FAISS.from_documents()` | `documents`, `embedding` | Store & index vectors |
| Retriever | `.as_retriever()` | `k` (number of results) | Fetch relevant docs |

---

> 📚 **Next Steps:**  
> - Try increasing `chunk_size` and see how it affects answer quality  
> - Swap `FAISS` for `Chroma` (persistent storage, no save/load needed)  
> - Try `MultiQueryRetriever` for better recall on complex questions  
> - Add memory to make it a full conversational RAG chatbot

# Assignment

## LangChain + Ollama Chat App 

A minimal chat application built with **LangChain**, **Ollama**, and the
`llama3.2:1b` model, demonstrating the classic `ChatPromptTemplate → LLM → StrOutputParser` chain pattern.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Install uv](#2-install-uv)
3. [Set Up the Project](#3-set-up-the-project)
4. [Pull the Ollama Model](#4-pull-the-ollama-model)
5. [Code Walkthrough](#5-code-walkthrough)
6. [Full Source Code](#6-full-source-code)
7. [Run the Application](#7-run-the-application)
8. [Expected Output](#8-expected-output)
9. [How It Works (Architecture)](#9-how-it-works-architecture)
10. [Common Errors and Fixes](#10-common-errors-and-fixes)
11. [Next Steps](#11-next-steps)

---

## 1. Prerequisites

| Requirement | Minimum version | Notes |
|-------------|----------------|-------|
| Python | 3.9+ | `python --version` |
| Ollama | latest | https://ollama.com |
| uv | latest | Replaces pip — fast Python package manager |

Make sure the **Ollama daemon is running** before you start:

```bash
ollama serve          # starts the local API on http://localhost:11434
```

---

## 2. Install uv

`uv` is a blazing-fast Python package and project manager written in Rust.
It replaces `pip`, `pip-tools`, `virtualenv`, and more — with a single binary.

**Windows (PowerShell):**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Verify the installation:
```bash
uv --version
```

> **Why uv instead of pip?**  
> `uv` resolves and installs packages **10–100x faster** than pip, manages
> virtual environments automatically, and produces reproducible lockfiles —
> all with a single tool.

---

## 3. Set Up the Project

### Option A — Virtual environment (recommended)

```bash
# Create a new project folder
mkdir langchain-ollama-chat 
cd langchain-ollama-chat

# Create a virtual environment
uv venv

# Activate it
.venv\Scripts\activate           # Windows

# Install dependencies
uv add langchain langchain-ollama langchain-core
```

`uv add` installs packages and records them in `pyproject.toml` automatically.

### Option B — Inline run (no setup needed)

```bash
uv run --with langchain --with langchain-ollama --with langchain-core \
    python langchain_ollama_chat.py
```

`uv run` creates a temporary environment, installs the packages, runs the script, then cleans up.

| Package | Purpose |
|---------|---------|
| `langchain` | Core framework and chain abstractions |
| `langchain-ollama` | `ChatOllama` integration for local models |
| `langchain-core` | `ChatPromptTemplate`, `StrOutputParser` |

---

## 4. Pull the Ollama Model

```bash
ollama pull llama3.2:1b
```

This downloads the **1-billion-parameter** Llama 3.2 model (~800 MB).
Verify it is available:

```bash
ollama list
```

---

## 5. Code Walkthrough

### Step 5.1 — Import Libraries

```python
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
```

- **`ChatOllama`** — wraps the local Ollama API as a LangChain chat model.
- **`ChatPromptTemplate`** — builds a prompt from a single template string with `{placeholders}`.
- **`StrOutputParser`** — converts the raw `AIMessage` object into a plain Python string.

---

### Step 5.2 — Initialise the LLM

```python
llm = ChatOllama(
    model="llama3.2:1b",
    temperature=0.7,
)
```

| Parameter | Value | Effect |
|-----------|-------|--------|
| `model` | `"llama3.2:1b"` | Selects the model pulled in Step 4 |
| `temperature` | `0.7` | Balances creativity vs. determinism (0 = fully deterministic, 1 = most creative) |

Ollama runs locally at `http://localhost:11434` by default — no API key needed.

---

### Step 5.3 — Build the Prompt Template

```python
prompt = ChatPromptTemplate.from_template(
    "You are a helpful, concise assistant. Answer clearly and briefly.\n\n"
    "User: {user_input}\n"
    "Assistant:"
)
```

- `from_template()` accepts a **single formatted string** with `{placeholder}` variables.
- The system instruction, user turn, and assistant cue are all embedded in one template.
- `{user_input}` is the only variable — it is substituted when the chain is invoked.
- Compare with `from_messages()`: that version uses explicit role tuples; `from_template()`
  is simpler when you want everything in one string.

---

### Step 5.4 — Add an Output Parser

```python
parser = StrOutputParser()
```

LangChain LLMs return an `AIMessage` object.
`StrOutputParser` extracts just the `.content` string so you receive plain text — no further unwrapping needed.

---

### Step 5.5 — Compose the Chain (LCEL)

```python
chain = prompt | llm | parser
```

This is LangChain’s **LCEL (LangChain Expression Language)** pipe syntax.
Each `|` passes the output of one component as the input of the next, similar to Unix pipes:

```
user_input  -->  prompt  -->  llm  -->  parser  -->  plain string
```

To invoke the chain:

```python
response = chain.invoke({"user_input": "What is the capital of France?"})
# response == "The capital of France is Paris."
```

---

### Step 5.6 — Interactive Chat Loop

```python
def chat():
    print("=" * 50)
    print("  LangChain + Ollama Chat  (llama3.2:1b)")
    print("  Type 'exit' or 'quit' to stop.")
    print("=" * 50)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        response = chain.invoke({"user_input": user_input})
        print(f"\nAssistant: {response}")
```

- Reads input from the terminal in a loop.
- Skips empty input gracefully.
- Handles `Ctrl+C` cleanly via `KeyboardInterrupt`.
- Passes each message through the chain and prints the plain-string response.
- Exits when the user types `exit` or `quit`.

> **Note:** This is a *single-turn* chat — the model does not retain memory of
> previous messages between turns. See [Next Steps](#11-next-steps) to add memory.

---

## 6. Full Source Code

Save this as **`langchain_ollama_chat.py`**:

```python
# langchain_ollama_chat.py
# A simple LangChain chat application using Ollama with llama3.2:1b

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ── 1. Model ────────────────────────────────────────────────────────────────────────────────
llm = ChatOllama(
    model="llama3.2:1b",
    temperature=0.7,
)

# ── 2. Prompt Template ────────────────────────────────────────────────────────────────────
prompt = ChatPromptTemplate.from_template(
    "You are a helpful, concise assistant. Answer clearly and briefly.\n\n"
    "User: {user_input}\n"
    "Assistant:"
)

# ── 3. Output Parser ──────────────────────────────────────────────────────────────────────
parser = StrOutputParser()

# ── 4. Chain  (prompt | llm | parser) ──────────────────────────────────────────────────────────
chain = prompt | llm | parser


# ── 5. Chat loop ───────────────────────────────────────────────────────────────────────────────
def chat():
    print("=" * 50)
    print("  LangChain + Ollama Chat  (llama3.2:1b)")
    print("  Type 'exit' or 'quit' to stop.")
    print("=" * 50)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        # Invoke the chain
        response = chain.invoke({"user_input": user_input})
        print(f"\nAssistant: {response}")


if __name__ == "__main__":
    chat()
```

---

## 7. Run the Application

**Terminal**
```bash
# Option A: activated uv venv
.venv\Scripts\activate           # Windows
python langchain_ollama_chat.py

# Option B: uv run (no activation needed)
uv run --with langchain --with langchain-ollama --with langchain-core \
    python langchain_ollama_chat.py
```

---

## 8. Expected Output

```
==================================================
  LangChain + Ollama Chat  (llama3.2:1b)
  Type 'exit' or 'quit' to stop.
==================================================

You: What is Python?

Assistant: Python is a high-level, interpreted programming language known
for its clean syntax and readability. It supports multiple paradigms
including object-oriented, functional, and procedural styles.

You: exit
Goodbye!
```

---

## 9. How It Works (Architecture)

```
Input string
    |
    v
ChatPromptTemplate.from_template()   <-- injects {user_input} into the template string
    |
    v
ChatOllama (llama3.2:1b)             <-- sends request to local Ollama API
    |                                    receives AIMessage back
    v
StrOutputParser                      <-- extracts .content string from AIMessage
    |
    v
Plain string response
```

The `|` operator in LCEL wires component I/O together:

| Step | Input | Output |
|------|-------|--------|
| `ChatPromptTemplate` | `dict` with `{user_input}` key | `ChatPromptValue` |
| `ChatOllama` | `ChatPromptValue` | `AIMessage` |
| `StrOutputParser` | `AIMessage` | `str` |

---

## 10. Common Errors and Fixes

**`Connection refused` / `httpx.ConnectError`** — Ollama is not running.
```bash
ollama serve
```

**`model 'llama3.2:1b' not found`** — Model has not been pulled yet.
```bash
ollama pull llama3.2:1b
```

**`ModuleNotFoundError: No module named 'langchain_ollama'`** — Missing packages.
```bash
uv add langchain langchain-ollama langchain-core
```

**`KeyError: 'user_input'`** — The key passed to `chain.invoke()` must exactly match
the placeholder name in the template string (e.g. `{user_input}`).

**Slow first response** — Normal. The model loads into memory on the first call; later turns are faster.

---

