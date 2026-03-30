from fastapi import FastAPI, File, HTTPException, UploadFile
import os
import shutil
from run_dpsk_ocr import DeepSeekOCRProcessor

app = FastAPI()

UPLOAD_FLODER = "uploads"

os.makedirs(UPLOAD_FLODER, exist_ok=True)

@app.post('/upload/')
async def create_upload_file(file: UploadFile = File(...)):

    allowed_types = ['image/jpeg', 'image/png', 'application/pdf']

    if file.content_type not in allowed_types:

        raise HTTPException(status_code=400, detail=f"Apenas esses tipos de arquivos são suportados: {allowed_types}")
    
    file_path = os.path.join(UPLOAD_FLODER, file.filename)

    with open(file_path, 'wb') as buffer:

        shutil.copyfileobj(file.file, buffer)

    DeepSeekOCRProcessor.main()

    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "message": "Dados processados com sucesso!"
    }

