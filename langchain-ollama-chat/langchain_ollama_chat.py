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