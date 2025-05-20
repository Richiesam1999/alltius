import os
from typing import Dict, Any, List
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.retrievers import MergerRetriever
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
import gradio as gr
import torch

# Constants for both vector stores
DB_PERSIST_DIR = "./db"
CHROMA_DB_PERSIST_DIR = "./chroma_db"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HEALTH_COLLECTION_NAME = "health_plans"
ANGELONE_COLLECTION_NAME = "angelone_support"

def get_embeddings():
    
    return HuggingFaceEmbeddings(
        model_name="nomic-ai/nomic-embed-text-v1",
        model_kwargs={'device': DEVICE, 'trust_remote_code': True},
        encode_kwargs={'normalize_embeddings': True}
    )

def get_llm():
    return Ollama(model="llama3.2", base_url="http://localhost:11434")

def load_vector_stores():
    embeddings = get_embeddings()
    stores = {}
    
    if os.path.exists(DB_PERSIST_DIR):
        try:
            health_store = Chroma(
                persist_directory=DB_PERSIST_DIR,
                embedding_function=embeddings,
                collection_name=HEALTH_COLLECTION_NAME
            )
            stores["health_plans"] = health_store
            print(f"Loaded health plans vector store from {DB_PERSIST_DIR}")
        except Exception as e:
            print(f"Error loading health plans vector store: {str(e)}")
    else:
        print(f"Health plans vector store not found at {DB_PERSIST_DIR}")
    
    # Check and load angelone support store
    if os.path.exists(CHROMA_DB_PERSIST_DIR):
        try:
            angelone_store = Chroma(
                persist_directory=CHROMA_DB_PERSIST_DIR,
                embedding_function=embeddings,
                collection_name=ANGELONE_COLLECTION_NAME
            )
            stores["angelone_support"] = angelone_store
            print(f"Loaded angelone support vector store from {CHROMA_DB_PERSIST_DIR}")
        except Exception as e:
            print(f"Error loading angelone support vector store: {str(e)}")
    else:
        print(f"Angelone support vector store not found at {CHROMA_DB_PERSIST_DIR}")
    
    return stores

def extract_plan_name(filename):
    """Extract the health plan name from the filename"""
    return os.path.splitext(filename)[0]

def answer_question(question: str) -> Dict[str, Any]:
    """Answer a question by checking both vector stores"""
    try:
        # Load both vector stores
        stores = load_vector_stores()
        
        if not stores:
            return {
                "answer": "No knowledge bases available. Please check the database directories.",
                "sources": []
            }
        
        llm = get_llm()
        
        # Create a common prompt template
        prompt_template = """
Answer the question based on the context provided. 
If the question asks about a specific plan or service, prioritize information from that source.
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
        
        # Try each store one by one
        best_result = None
        best_score = 0  
        
        for store_name, vector_store in stores.items():
            try:
                print(f"Querying {store_name} store...")
                
                # Configure retriever for this store
                retriever = vector_store.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 3}
                )
                
                # Create QA chain for this store
                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=retriever,
                    return_source_documents=True,
                    chain_type_kwargs={"prompt": PROMPT}
                )
                
                # Query this store
                result = qa_chain({"query": question})
                
                # Check if we got a meaningful answer
                answer = result.get("result", "")
                source_docs = result.get("source_documents", [])
                
            
                if (not answer or 
                    answer.lower().strip() in [
                        "i don't know", 
                        "i don't have that information in my knowledge base",
                        "i don't have that information"
                    ] or
                    "don't have" in answer.lower() or
                    "no information" in answer.lower()):
                    continue
                
                # Calculate a simple relevance score based on source docs
                # More relevant docs = higher score
                score = len(source_docs)
                
                # If this is the first valid result or better than previous, keep it
                if best_result is None or score > best_score:
                    # Format sources based on which store we're using
                    sources = []
                    for doc in source_docs:
                        if store_name == "health_plans":
                            # For health plans, include plan name and file source
                            file_source = doc.metadata.get("source", "Unknown source")
                            plan_name = doc.metadata.get("plan_name", extract_plan_name(file_source))
                            source_entry = f"{plan_name} ({file_source})"
                        else:
                            # For angelone support, just use the source
                            source_entry = doc.metadata.get("source", "Unknown source")
                        
                        if source_entry not in sources:
                            sources.append(source_entry)
                    
                    best_result = {
                        "answer": answer,
                        "sources": sources,
                        "store": store_name
                    }
                    best_score = score
                
            except Exception as e:
                print(f"Error querying {store_name} store: {str(e)}")
        
        # If we found a result in any store, return it
        if best_result:
            print(f"Found best answer in {best_result['store']} store")
            return {
                "answer": best_result["answer"],
                "sources": best_result["sources"]
            }
        
        # If no results from any store, return not found message
        return {
            "answer": "I don't have information about that in my knowledge base.",
            "sources": []
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
    print("Loading vector stores...")
    stores = load_vector_stores()
    if not stores:
        print("Warning: No vector stores were found. Please check the directories.")
    else:
        print(f"Successfully loaded {len(stores)} vector stores.")

    demo.launch(server_name="0.0.0.0", server_port=7860)