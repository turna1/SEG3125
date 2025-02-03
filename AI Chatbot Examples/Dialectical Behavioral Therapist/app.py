import os
import gradio as gr
from google.generativeai import GenerativeModel, configure, types
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from typing import List, Tuple  # Make sure to import List and Tuple

# Set up the Google API for the Gemini model
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
configure(api_key=GOOGLE_API_KEY)

# Placeholder for the app's state
class MyApp:
    def __init__(self) -> None:
        self.documents = []
        self.embeddings = None
        self.index = None
        self.load_pdf("THEDIA1.pdf")
        self.build_vector_db()

    def load_pdf(self, file_path: str) -> None:
        """Extracts text from a PDF file and stores it in the app's documents."""
        doc = fitz.open(file_path)
        self.documents = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            self.documents.append({"page": page_num + 1, "content": text})
        print("PDF processed successfully!")

    def build_vector_db(self) -> None:
        """Builds a vector database using the content of the PDF."""
        model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = model.encode([doc["content"] for doc in self.documents], show_progress_bar=True)
        self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
        self.index.add(np.array(self.embeddings))
        print("Vector database built successfully!")

    def search_documents(self, query: str, k: int = 3) -> List[str]:
        """Searches for relevant documents using vector similarity."""
        model = SentenceTransformer('all-MiniLM-L6-v2')
        query_embedding = model.encode([query], show_progress_bar=False)
        D, I = self.index.search(np.array(query_embedding), k)
        results = [self.documents[i]["content"] for i in I[0]]
        return results if results else ["No relevant documents found."]

app = MyApp()

def respond(message: str, history: List[Tuple[str, str]]):
    system_message = (
        "You are a supportive and empathetic Dialectical Behaviour Therapist assistant. "
        "You politely guide users through DBT exercises based on the given DBT book. "
        "You must say one thing at a time and ask follow-up questions to continue the chat."
    )
    messages = [{"role": "system", "content": system_message}]

    for val in history:
        if val[0]:
            messages.append({"role": "user", "content": val[0]})
        if val[1]:
            messages.append({"role": "assistant", "content": val[1]})

    messages.append({"role": "user", "content": message})

    # RAG - Retrieve relevant documents if the query suggests exercises or specific information
    if any(
        keyword in message.lower()
        for keyword in ["exercise", "technique", "information", "guide", "help", "how to"]
    ):
        retrieved_docs = app.search_documents(message)
        context = "\n".join(retrieved_docs)
        if context.strip():
            messages.append({"role": "system", "content": "Relevant documents: " + context})

    # Generate response using the generative model
    model = GenerativeModel("gemini-1.5-pro-latest")
    generation_config = types.GenerationConfig(
        temperature=0.7,
        max_output_tokens=1024,
    )
    
    try:
        response = model.generate_content([message], generation_config=generation_config)
        # Properly access the response content
        response_content = response.text if hasattr(response, "text") else "No response generated."
    except Exception as e:
        response_content = f"An error occurred while generating the response: {str(e)}"

    # Append the message and generated response to the chat history
    history.append((message, response_content))
    return history, ""

def old_respond(message: str, history: List[Tuple[str, str]]):
    system_message = "You are a supportive and empathetic Dialectical Behaviour Therapist assistant. You politely guide users through DBT exercises based on the given DBT book. You must say one thing at a time and ask follow-up questions to continue the chat."
    messages = [{"role": "system", "content": system_message}]

    for val in history:
        if val[0]:
            messages.append({"role": "user", "content": val[0]})
        if val[1]:
            messages.append({"role": "assistant", "content": val[1]})

    messages.append({"role": "user", "content": message})

    # RAG - Retrieve relevant documents if the query suggests exercises or specific information
    if any(keyword in message.lower() for keyword in ["exercise", "technique", "information", "guide", "help", "how to"]):
        retrieved_docs = app.search_documents(message)
        context = "\n".join(retrieved_docs)
        if context.strip():
            messages.append({"role": "system", "content": "Relevant documents: " + context})

    model = GenerativeModel("gemini-1.5-pro-latest")
    generation_config = types.GenerationConfig(
        temperature=0.7,
        max_output_tokens=1024
    )
    response = model.generate_content([message], generation_config=generation_config)

    response_content = response[0].text if response else "No response generated."
    history.append((message, response_content))
    return history, ""

with gr.Blocks(theme=gr.themes.Glass(primary_hue = "violet")) as demo:
    gr.Markdown("# 🧘‍♀️ **Dialectical Behaviour Therapy**")
    gr.Markdown(
        "‼️Disclaimer: This chatbot is based on a DBT exercise book that is publicly available. "
        "We are not medical practitioners, and the use of this chatbot is at your own responsibility."
    )

    chatbot = gr.Chatbot()

    with gr.Row():
        txt_input = gr.Textbox(
            show_label=False,
            placeholder="",
            lines=1
        )
        submit_btn = gr.Button("Submit", scale=1)
        refresh_btn = gr.Button("Refresh Chat", scale=1, variant="secondary")

    example_questions = [
        ["What are some techniques to handle distressing situations?"],
        ["How does DBT help with emotional regulation?"],
        ["Can you give me an example of an interpersonal effectiveness skill?"],
        ["I want to practice mindfulness. Can you help me?"],
        ["I want to practice distraction techniques. What can I do?"],
        ["How do I plan self-accommodation?"],
        ["What are some distress tolerance skills?"],
        ["Can you help me with emotional regulation techniques?"],
        ["How can I improve my interpersonal effectiveness?"],
        ["What are some ways to cope with stress using DBT?"],
        ["Can you guide me through a grounding exercise?"]
    ]

    gr.Examples(examples=example_questions, inputs=[txt_input])

    submit_btn.click(fn=respond, inputs=[txt_input, chatbot], outputs=[chatbot, txt_input])
    refresh_btn.click(lambda: [], None, chatbot)

if __name__ == "__main__":
    demo.launch()
