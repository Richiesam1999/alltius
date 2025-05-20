import os
from typing import Dict, Any, List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
import gradio as gr
import torch


PERSIST_DIR = "./chroma_db"
TEXT_FILES_DIR = "./"  # Directory where your text files are stored
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COLLECTION_NAME = "angelone_support"

def get_embeddings():
    """Initialize the embedding model"""
    return HuggingFaceEmbeddings(
        model_name="nomic-ai/nomic-embed-text-v1",
        model_kwargs={'device': DEVICE, 'trust_remote_code': True},
        encode_kwargs={'normalize_embeddings': True}
    )

def get_llm():
    return Ollama(model="llama3.2", base_url="http://localhost:11434")

def process_text_files():
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    
    documents = []
    for filename in os.listdir(TEXT_FILES_DIR):
        if filename.startswith("angelone_") and filename.endswith(".txt"):
            with open(os.path.join(TEXT_FILES_DIR, filename), 'r', encoding='utf-8') as f:
                content = f.read()
                # Extract source from first line
                first_line = content.split('\n')[0]
                source = first_line.replace('===', '').strip()
                # Remove the source line from content
                content = '\n'.join(content.split('\n')[1:])
                
                chunks = text_splitter.create_documents(
                    [content],
                    metadatas=[{"source": source}]
                )
                documents.extend(chunks)
    
    return documents

def initialize_vector_store():
    """Initialize or load the Chroma vector store"""
    embeddings = get_embeddings()
    
    if os.path.exists(PERSIST_DIR):
        return Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=embeddings,
            collection_name=COLLECTION_NAME
        )
    else:
        documents = process_text_files()
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=PERSIST_DIR,
            collection_name=COLLECTION_NAME
        )
        vector_store.persist()
        return vector_store

def answer_question(question: str) -> Dict[str, Any]:
    """Answer a question using the RAG system"""
    try:
        vector_store = initialize_vector_store()
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )
        
        llm = get_llm()
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
        
        result = qa_chain({"query": question})
        
        # Format the response with sources
        answer = result["result"]
        sources = list(set([doc.metadata["source"] for doc in result["source_documents"]]))
        
        if not answer or answer.lower() == "i don't know":
            return {
                "answer": "No answer present in the knowledge base.",
                "sources": []
            }
        
        return {
            "answer": answer,
            "sources": sources
        }
    except Exception as e:
        return {
            "answer": f"Error processing your question: {str(e)}",
            "sources": []
        }

def format_response(response: Dict[str, Any]) -> str:
    """Format the response for Gradio display"""
    answer = response["answer"]
    sources = response["sources"]
    
    if sources:
        sources_text = "\n\nSources:\n- " + "\n- ".join(sources)
        return answer + sources_text
    return answer

def chat_interface(question: str, history: List[List[str]]):
    response = answer_question(question)
    return format_response(response)

# Gradio Interface
with gr.Blocks(title="Alltius Support Chatbot") as demo:
    gr.Markdown("# Alltius Support Chatbot")
    gr.Markdown("Ask questions about Alltius services and support")
    
    chatbot = gr.Chatbot(height=500)
    msg = gr.Textbox(label="Your Question")
    clear = gr.Button("Clear")
    
    def respond(message, chat_history):
        response = answer_question(message)
        formatted_response = format_response(response)
        chat_history.append((message, formatted_response))
        return "", chat_history
    
    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    clear.click(lambda: None, None, chatbot, queue=False)

if __name__ == "__main__":
    # Initialize vector store first
    print("Initializing vector store...")
    initialize_vector_store()
    print("Vector store ready!")
    
    # Launch Gradio interface
    demo.launch(server_name="0.0.0.0", server_port=7860)