"""
RVC Real-Time Voice Conversion System
RunPod Serverless Load Balancer Deployment
"""

import os
import asyncio
import base64
import json
import uuid
import time
from typing import Optional, Dict, Any
from datetime import datetime

import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

# Import RVC modules
from rvc_inference import RVCInference
from rvc_trainer import RVCTrainer

# ============================================
# Configuration
# ============================================

MODEL_CACHE_DIR = "/app/models"
TRAINING_CACHE_DIR = "/app/training"
LOG_DIR = "/app/logs"

os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
os.makedirs(TRAINING_CACHE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Global model cache
model_cache: Dict[str, RVCInference] = {}
training_jobs: Dict[str, Dict[str, Any]] = {}

# Get API key from environment
API_KEY = os.environ.get("RUNPOD_API_KEY", "default-dev-key-please-change-in-production")

# ============================================
# Pydantic Models
# ============================================

class TrainRequest(BaseModel):
    """Training request - flat structure as required"""
    user_id: str = Field(..., description="Unique user identifier")
    voice_name: str = Field(..., description="Name for the voice model")
    audio: str = Field(..., description="Base64 encoded audio file")

class TrainResponse(BaseModel):
    success: bool
    message: str
    model_id: Optional[str] = None
    training_time: Optional[float] = None

class VoiceConversionRequest(BaseModel):
    user_id: str
    voice_name: str
    audio_chunk: str  # Base64 encoded PCM16 audio

class VoiceConversionResponse(BaseModel):
    audio_chunk: str  # Base64 encoded converted PCM16 audio
    processing_time_ms: float

# ============================================
# Security
# ============================================

security = HTTPBearer()

def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify Bearer token for HTTP requests"""
    if credentials.credentials != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials

async def verify_websocket_token(websocket: WebSocket, token: Optional[str] = None):
    """Verify token from query parameter for WebSocket"""
    if not token:
        await websocket.close(code=1008, reason="Missing authentication token")
        return False
    
    if token != API_KEY:
        await websocket.close(code=1008, reason="Invalid authentication token")
        return False
    
    return True

# ============================================
# FastAPI App with Lifespan Management
# ============================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    print("🚀 Starting RVC Voice Cloning Server...")
    print(f"📁 Model cache: {MODEL_CACHE_DIR}")
    print(f"🔑 API Key configured: {'Yes' if API_KEY != 'default-dev-key-please-change-in-production' else 'Using default (insecure!)'}")
    yield
    print("🛑 Shutting down RVC Voice Cloning Server...")
    # Cleanup models
    for model in model_cache.values():
        model.cleanup()

app = FastAPI(
    title="RVC Voice Cloning API",
    description="Real-time voice conversion with RVC",
    version="1.0.0",
    lifespan=lifespan
)

# CORS Middleware - Allow all origins for RunPod compatibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# ============================================
# Helper Functions
# ============================================

def decode_base64_audio(base64_str: str) -> bytes:
    """Decode base64 string to bytes"""
    return base64.b64decode(base64_str)

def encode_base64_audio(audio_bytes: bytes) -> str:
    """Encode bytes to base64 string"""
    return base64.b64encode(audio_bytes).decode('utf-8')

def pcm16_to_float32(pcm16_bytes: bytes) -> np.ndarray:
    """Convert PCM16 bytes to float32 numpy array"""
    pcm16 = np.frombuffer(pcm16_bytes, dtype=np.int16)
    return pcm16.astype(np.float32) / 32768.0

def float32_to_pcm16(float32_array: np.ndarray) -> bytes:
    """Convert float32 numpy array to PCM16 bytes"""
    # Clip to [-1, 1] range
    clipped = np.clip(float32_array, -1.0, 1.0)
    pcm16 = (clipped * 32767).astype(np.int16)
    return pcm16.tobytes()

def get_model_key(user_id: str, voice_name: str) -> str:
    """Generate unique model key"""
    return f"{user_id}_{voice_name}".replace(" ", "_").lower()

# ============================================
# Training Endpoint
# ============================================

@app.post("/train", response_model=TrainResponse)
async def train_voice_model(
    request: TrainRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Train a new RVC voice model from uploaded audio sample.
    Accepts flat JSON payload: user_id, voice_name, audio
    """
    start_time = time.time()
    
    try:
        model_key = get_model_key(request.user_id, request.voice_name)
        
        # Check if model already exists
        if model_key in model_cache:
            return TrainResponse(
                success=True,
                message=f"Model '{request.voice_name}' already loaded",
                model_id=model_key,
                training_time=0
            )
        
        # Decode audio
        audio_bytes = decode_base64_audio(request.audio)
        
        # Save audio file temporarily
        temp_audio_path = os.path.join(TRAINING_CACHE_DIR, f"{model_key}_input.wav")
        with open(temp_audio_path, "wb") as f:
            f.write(audio_bytes)
        
        # Initialize trainer
        trainer = RVCTrainer(
            model_key=model_key,
            cache_dir=MODEL_CACHE_DIR,
            temp_dir=TRAINING_CACHE_DIR
        )
        
        # Train the model (this would be async in production)
        # For demo, we'll simulate training or implement actual RVC training
        training_success = await trainer.train_from_audio(
            audio_path=temp_audio_path,
            voice_name=request.voice_name
        )
        
        if not training_success:
            raise Exception("Training failed")
        
        # Load the trained model
        inference = RVCInference(
            model_key=model_key,
            model_path=os.path.join(MODEL_CACHE_DIR, f"{model_key}.pth"),
            config_path=os.path.join(MODEL_CACHE_DIR, f"{model_key}.json")
        )
        
        inference.load()
        model_cache[model_key] = inference
        
        training_time = time.time() - start_time
        
        return TrainResponse(
            success=True,
            message=f"Voice '{request.voice_name}' trained successfully in {training_time:.1f}s",
            model_id=model_key,
            training_time=training_time
        )
        
    except Exception as e:
        print(f"Training error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================
# Health Check
# ============================================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": len(model_cache),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/models")
async def list_models(api_key: str = Depends(verify_api_key)):
    """List all loaded models"""
    return {
        "models": list(model_cache.keys()),
        "count": len(model_cache)
    }

# ============================================
# WebSocket Endpoint for Real-time Conversion
# ============================================

@app.websocket("/ws")
async def websocket_voice_conversion(websocket: WebSocket):
    """
    WebSocket endpoint for real-time voice conversion.
    Token must be passed as query parameter: ?token=<API_KEY>
    """
    # Extract token from query parameters
    token = websocket.query_params.get("token")
    
    # Verify token
    if not await verify_websocket_token(websocket, token):
        return
    
    await websocket.accept()
    print(f"✅ WebSocket connected: {websocket.client}")
    
    current_model_key = None
    current_model = None
    
    try:
        while True:
            # Receive message
            data = await websocket.receive_text()
            
            try:
                message = json.loads(data)
                user_id = message.get("user_id")
                voice_name = message.get("voice_name")
                audio_chunk_b64 = message.get("audio_chunk")
                
                if not all([user_id, voice_name, audio_chunk_b64]):
                    await websocket.send_json({
                        "error": "Missing required fields: user_id, voice_name, audio_chunk"
                    })
                    continue
                
                # Get or load model
                model_key = get_model_key(user_id, voice_name)
                
                if model_key != current_model_key:
                    if model_key in model_cache:
                        current_model = model_cache[model_key]
                        current_model_key = model_key
                        print(f"📦 Using cached model: {model_key}")
                    else:
                        # Model not found - send error
                        await websocket.send_json({
                            "error": f"Model for voice '{voice_name}' not found. Please train it first using /train endpoint."
                        })
                        continue
                
                if current_model is None:
                    await websocket.send_json({
                        "error": "No model loaded. Please train a voice model first."
                    })
                    continue
                
                # Process audio chunk
                start_process = time.time()
                
                # Decode audio
                audio_bytes = decode_base64_audio(audio_chunk_b64)
                audio_float32 = pcm16_to_float32(audio_bytes)
                
                # Perform voice conversion
                converted_audio = await current_model.convert(audio_float32)
                
                # Convert back to PCM16
                converted_bytes = float32_to_pcm16(converted_audio)
                converted_b64 = encode_base64_audio(converted_bytes)
                
                processing_time = (time.time() - start_process) * 1000
                
                # Send response
                await websocket.send_json({
                    "audio_chunk": converted_b64,
                    "processing_time_ms": round(processing_time, 2)
                })
                
            except json.JSONDecodeError:
                await websocket.send_json({"error": "Invalid JSON format"})
            except Exception as e:
                print(f"Processing error: {e}")
                await websocket.send_json({"error": f"Processing error: {str(e)}"})
                
    except WebSocketDisconnect:
        print(f"🔌 WebSocket disconnected: {websocket.client}")
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        # Cleanup
        pass

# ============================================
# Main Entry Point
# ============================================

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        workers=1
    )
