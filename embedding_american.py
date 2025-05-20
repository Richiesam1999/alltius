import os
from typing import Dict, Any, List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
import gradio as gr
import torch


PERSIST_DIR = "./db"
TEXT_FILES_DIR = "./med_docs"  # Directory where your health plan text files are stored
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COLLECTION_NAME = "health_plans"

def get_embeddings():
    """Initialize the embedding model"""
    return HuggingFaceEmbeddings(
        model_name="nomic-ai/nomic-embed-text-v1",
        model_kwargs={'device': DEVICE, 'trust_remote_code': True},
        encode_kwargs={'normalize_embeddings': True}
    )

def get_llm():
    return Ollama(model="llama3.2", base_url="http://localhost:11434")

def extract_plan_name(filename):
    """Extract the health plan name from the filename"""
    # Assuming the filename format is like "PSM Health Plan.txt" or similar
    # Remove file extension and return the plan name
    return os.path.splitext(filename)[0]

def process_text_files():
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    
    documents = []
    
    # Check if the directory exists
    if not os.path.exists(TEXT_FILES_DIR):
        print(f"Directory not found: {TEXT_FILES_DIR}")
        return documents
    
    # Process all text files in the directory
    for filename in os.listdir(TEXT_FILES_DIR):
        if filename.endswith(".txt"):
            file_path = os.path.join(TEXT_FILES_DIR, filename)
            
            # Extract plan name from filename
            plan_name = extract_plan_name(filename)
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                    # Create document chunks with metadata containing both filename and plan name
                    chunks = text_splitter.create_documents(
                        [content],
                        metadatas=[{
                            "source": filename,
                            "plan_name": plan_name
                        }] * len(text_splitter.split_text(content))
                    )
                    documents.extend(chunks)
                    print(f"Processed {filename}, created {len(chunks)} chunks")
            except Exception as e:
                print(f"Error processing file {filename}: {str(e)}")
    
    return documents

def initialize_vector_store(force_rebuild=False):
    """Initialize or load the Chroma vector store"""
    embeddings = get_embeddings()
    
    # If force_rebuild is True or the persist directory doesn't exist, create a new vector store
    if force_rebuild or not os.path.exists(PERSIST_DIR):
        # Ensure the directory exists
        os.makedirs(PERSIST_DIR, exist_ok=True)
        
        # Process text files
        documents = process_text_files()
        if not documents:
            print("No documents processed. Please check the text files directory.")
            return None
        
        # Create and persist the vector store
        print(f"Creating new vector store with {len(documents)} documents...")
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=PERSIST_DIR,
            collection_name=COLLECTION_NAME
        )
        vector_store.persist()
        return vector_store
    else:
        # Load existing vector store
        print("Loading existing vector store...")
        return Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=embeddings,
            collection_name=COLLECTION_NAME
        )

def answer_question(question: str) -> Dict[str, Any]:
    """Answer a question using the RAG system"""
    try:
        vector_store = initialize_vector_store()
        if not vector_store:
            return {
                "answer": "Failed to initialize the knowledge base. Please check the logs.",
                "sources": []
            }
        
        # Check if the question contains plan name references
        # This will help prioritize documents from specific plans
        plan_specific_retriever = None
        
        # Configure retriever for better results
        retriever = vector_store.as_retriever(
            search_type="mmr",  # Use Maximum Marginal Relevance for better diversity
            search_kwargs={"k": 5, "fetch_k": 10}  # Fetch more, then filter for diversity
        )
        
        llm = get_llm()
        
        # Configure retrieval chain
        prompt_template = """
Answer the question based on the context provided. 
If the question asks about a specific health plan, prioritize information from that plan.
If the answer is not in the context, say "I don't have that information in my knowledge base" instead of making up an answer.

Context:
{context}

Question:
{question}

Answer:
"""
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
        
        result = qa_chain({"query": question})
        
        # Extract the answer and source documents
        answer = result["result"]
        source_docs = result["source_documents"]
        
        # Format the sources with both filename and plan name
        sources = []
        for doc in source_docs:
            file_source = doc.metadata.get("source", "Unknown source")
            plan_name = doc.metadata.get("plan_name", extract_plan_name(file_source))
            source_entry = f"{plan_name} ({file_source})"
            if source_entry not in sources:
                sources.append(source_entry)
        
        if not answer or answer.lower() in ["i don't know", "i don't have that information in my knowledge base"]:
            return {
                "answer": "The information is not present in the knowledge base.",
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
    rebuild_db = gr.Button("Rebuild Knowledge Base")
    
    def respond(message, chat_history):
        response = answer_question(message)
        formatted_response = format_response(response)
        chat_history.append((message, formatted_response))
        return "", chat_history
    
    def rebuild_knowledge_base():
        initialize_vector_store(force_rebuild=True)
        return "Knowledge base rebuilt successfully!"
    
    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    clear.click(lambda: None, None, chatbot, queue=False)
    rebuild_db.click(rebuild_knowledge_base, None, None)

if __name__ == "__main__":
    # Initialize vector store first
    print("Initializing vector store...")
    initialize_vector_store()
    print("Vector store ready!")
    
    # Launch Gradio interface
    demo.launch(server_name="0.0.0.0", server_port=7860)