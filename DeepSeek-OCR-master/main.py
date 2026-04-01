import os
import shutil
import torch
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, HTTPException, UploadFile
from transformers import AutoModel, AutoTokenizer
from run_dpsk_ocr import DeepSeekOCRProcessor

# Restrict PyTorch to only use GPU index 0
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Enable expandable memory segments to reduce GPU memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Folder where uploaded files will be temporarily stored
UPLOAD_FOLDER = "uploads"

# Create the uploads folder if it does not exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# HuggingFace model identifier
model_name = 'deepseek-ai/DeepSeek-OCR'

# Global dictionary to hold the model and tokenizer across requests
state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    
    # Load model and tokenizer once when the server starts
    print("--- Carregando modelo e tokenizer ---")

    # Load the tokenizer for the OCR model
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Load the model in bfloat16 precision, distributed across GPU and CPU
    model = AutoModel.from_pretrained(

        model_name,
        # Allow execution of custom code from the model repository
        trust_remote_code=True,

        # Use reduced precision to save GPU memory
        torch_dtype=torch.bfloat16,

        # Automatically split layers between GPU and CPU
        device_map="auto",

        # Move intermediate buffers to CPU when GPU memory is tight
        offload_buffers=True,

        # Set memory limits per device to avoid OOM errors
        max_memory={0: "4.5GiB", "cpu": "16GiB"}
    )

    # Set model to inference mode, disabling dropout and gradient tracking
    model.eval()

    # Store model and tokenizer globally so all requests can reuse them
    state["model"] = model
    state["tokenizer"] = tokenizer

    print("--- Modelo carregado e pronto ---")

    # Server stays alive here until shutdown is triggered
    yield

    # Release model and tokenizer from memory when the server shuts down
    state.clear()


# Create the FastAPI app and register the lifespan handler
app = FastAPI(lifespan=lifespan)


def _cleanup_folder(folder_path: str):

    # Loop through every item in the given folder
    for item in os.listdir(folder_path):

        # Build the full path for the current item
        item_path = os.path.join(folder_path, item)

        try:

            # Remove the item if it is a regular file
            if os.path.isfile(item_path):
                os.remove(item_path)

            # Remove the item and all its contents if it is a directory
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)

        except Exception as e:
            # Log the error and continue cleaning remaining items
            print(f"Erro ao deletar: {item_path}: {e}")


@app.post('/upload/')
async def create_upload_file(file: UploadFile = File(...)):

    # Define accepted MIME types for uploaded files
    allowed_types = ['image/jpeg', 'image/png', 'image/jpg', 'application/pdf']

    # Reject the file if its type is not in the allowed list
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail=f"Tipos suportados: {allowed_types}")

    # Build the destination path inside the uploads folder
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)

    # Write the uploaded file content to disk in binary mode
    with open(file_path, 'wb') as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:

        # Run the OCR processor using the preloaded model and tokenizer
        result_ocr = DeepSeekOCRProcessor.main(
            model=state["model"],
            tokenizer=state["tokenizer"]
        )

        # Return the structured OCR result as the HTTP response
        return {
            "filename": file.filename,
            "content_data": result_ocr,
            "message": "Dados processados com sucesso!"
        }

    except Exception as e:

        # Return HTTP 500 with the error details if processing fails
        raise HTTPException(status_code=500, detail=f"Erro no processamento: {str(e)}")

    finally:

        # Always clean the uploads folder after each request, success or failure
        _cleanup_folder(UPLOAD_FOLDER)