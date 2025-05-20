import os
import gradio as gr
from main import load_embeddings, answer_query, process_documents, create_embeddings

# Check if database exists and load or create it
def setup_database():
    if os.path.exists("db") and os.listdir("db"):
        print("Loading existing database...")
        db = load_embeddings()
    else:
        print("Creating new database...")
        os.makedirs("docs", exist_ok=True)
        print("Please add document files to the 'docs' directory before proceeding.")
        print("No documents found. Database will be created when documents are added.")
        db = None
    return db

# Function to handle file uploads and update the database
def upload_files(files):
    # Create docs directory if it doesn't exist
    os.makedirs("docs", exist_ok=True)
    
    # Save uploaded files to docs directory
    for file in files:
        file_path = file.name
        file_name = os.path.basename(file_path)
        destination = os.path.join("docs", file_name)
        
        # Copy the file
        with open(file_path, "rb") as src, open(destination, "wb") as dst:
            dst.write(src.read())
    
    # Process documents and create/update embeddings
    chunks = process_documents()
    db = create_embeddings(chunks)
    
    return f"Successfully processed {len(files)} files and updated the database."

# Function to handle queries
def process_query(query, num_results):
    db = setup_database()
    if db is None:
        return "No documents have been processed yet. Please upload documents first."
    
    return answer_query(query, db)

# Create the Gradio interface
def create_interface():
    with gr.Blocks(title="Document RAG System") as demo:
        gr.Markdown("# Document RAG System")
        gr.Markdown("Upload PDF and DOCX files, then ask questions about them.")
        
        with gr.Tab("Upload Documents"):
            file_input = gr.File(label="Upload PDF or DOCX files", file_count="multiple")
            upload_button = gr.Button("Process Files")
            upload_output = gr.Textbox(label="Processing Result")
            
            upload_button.click(
                fn=upload_files,
                inputs=[file_input],
                outputs=[upload_output]
            )
        
        with gr.Tab("Query Documents"):
            query_input = gr.Textbox(label="Your Question", placeholder="What is the overall deductible?")
            num_results = gr.Slider(minimum=1, maximum=10, value=3, step=1, label="Number of Results")
            submit_button = gr.Button("Submit")
            response_output = gr.Textbox(label="Response with Sources", lines=15)
            
            submit_button.click(
                fn=process_query,
                inputs=[query_input, num_results],
                outputs=[response_output]
            )
    
    return demo

if __name__ == "__main__":
    # Setup the database
    db = setup_database()
    
    # Launch the interface
    demo = create_interface()
    demo.launch(share=True)