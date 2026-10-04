import io
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse, StreamingResponse
from PIL import Image
import uvicorn
import os

app = FastAPI()
MODELS_DIR = "models"
models = {}

def get_model_path(filename):
    return os.path.join(MODELS_DIR, filename)

def load_models():
    if not models:
        try:
            models['universal'] = ort.InferenceSession(get_model_path("universal_autoencoder.onnx"))
            models['classifier'] = ort.InferenceSession(get_model_path("corruption_classifier.onnx"))
            models['spec1'] = ort.InferenceSession(get_model_path("specialist_1.onnx"))
            models['spec2'] = ort.InferenceSession(get_model_path("specialist_2.onnx"))
            models['spec3'] = ort.InferenceSession(get_model_path("specialist_3.onnx"))
            models['moe'] = ort.InferenceSession(get_model_path("soft_moe_system.onnx"))
            print("Restoration models loaded.")
        except Exception as e:
            print(f"Error loading restoration models: {e}")
        try:
            # Load the new CGAN model!
            models['cgan'] = ort.InferenceSession(get_model_path("cgan_generator.onnx"))
            print("CGAN model loaded.")
        except Exception as e:
            print(f"CGAN model not found yet: {e}")

def preprocess(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((128, 128))
    img_np = np.array(img).astype(np.float32) / 255.0
    img_np = np.transpose(img_np, (2, 0, 1))
    img_np = np.expand_dims(img_np, axis=0)
    return img_np, img

def postprocess(tensor):
    tensor = np.squeeze(tensor, axis=0)
    tensor = np.transpose(tensor, (1, 2, 0))
    tensor = np.clip(tensor, 0.0, 1.0) * 255.0
    img = Image.fromarray(tensor.astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf

@app.post("/predict")
async def predict(file: UploadFile = File(...), model_type: str = Form(...)):
    load_models()
    img_bytes = await file.read()
    input_tensor, _ = preprocess(img_bytes)
    
    if model_type == "universal":
        out = models['universal'].run(None, {models['universal'].get_inputs()[0].name: input_tensor})[0]
    elif model_type == "hard":
        logits = models['classifier'].run(None, {models['classifier'].get_inputs()[0].name: input_tensor})[0]
        pred_class = np.argmax(logits, axis=1)[0]
        if pred_class == 0:   out = input_tensor
        elif pred_class == 1: out = models['spec1'].run(None, {models['spec1'].get_inputs()[0].name: input_tensor})[0]
        elif pred_class == 2: out = models['spec2'].run(None, {models['spec2'].get_inputs()[0].name: input_tensor})[0]
        elif pred_class == 3: out = models['spec3'].run(None, {models['spec3'].get_inputs()[0].name: input_tensor})[0]
    elif model_type == "soft":
        out = models['moe'].run(None, {models['moe'].get_inputs()[0].name: input_tensor})[0]
        
    return StreamingResponse(postprocess(out), media_type="image/png")

@app.post("/predict_sketch")
async def predict_sketch(file: UploadFile = File(...), style: int = Form(...)):
    load_models()
    img_bytes = await file.read()
    
    # 1. GAN Preprocessing: Range must be [-1, 1] because of Tanh activation!
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB").resize((128, 128))
    img_np = np.array(img).astype(np.float32) / 255.0
    img_np = (img_np - 0.5) / 0.5
    img_np = np.transpose(img_np, (2, 0, 1))
    photo_tensor = np.expand_dims(img_np, axis=0)
    
    style_tensor = np.array([style], dtype=np.int64)
    
    # 2. Inference
    out = models['cgan'].run(None, {'photo': photo_tensor, 'style': style_tensor})[0]
    
    # 3. GAN Postprocessing: Range [-1, 1] back to [0, 255]
    tensor = np.squeeze(out, axis=0)
    tensor = np.transpose(tensor, (1, 2, 0))
    tensor = (tensor * 0.5) + 0.5
    tensor = np.clip(tensor, 0.0, 1.0) * 255.0
    
    img_out = Image.fromarray(tensor.astype(np.uint8))
    buf = io.BytesIO()
    img_out.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")

@app.get("/")
async def index():
    with open("index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
