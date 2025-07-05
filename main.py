from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSerializable
from langchain_core.documents import Document
from typing import Dict
from vector import retriever  

# Initialize Ollama LLM
model = OllamaLLM(model="llama3")  

# Prompt template
template = """
You are Jarvis, an expert AI assistant trained to answer questions based on internal knowledge and customer feedback.

Here are some relevant reviews and facts:
{reviews}

Now answer this question clearly and helpfully:
{question}
"""

# Create chain: prompt → LLM
prompt = ChatPromptTemplate.from_template(template)
chain: RunnableSerializable[Dict[str, str], str] = prompt | model

# Start asking questions
if __name__ == "__main__":
    print("🧠 Jarvis AI is now running! Type 'q' to quit.")
    while True:
        print("\n-------------------------------")
        question = input("Ask your question: ").strip()

        if question.lower() == "q":
            print("Goodbye! Shutting down Jarvis.")
            break

        try:
            # Convert documents to readable text
            docs = retriever.invoke(question)
            reviews_text = "\n".join([doc.page_content for doc in docs])
            
            # Call the LLM
            result = chain.invoke({"reviews": reviews_text, "question": question})
            print("\n🤖 Jarvis says:\n" + result)
        except Exception as e:
            print("⚠️ Error:", e)
