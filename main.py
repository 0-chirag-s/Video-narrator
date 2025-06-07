from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import torch
import torchvision.transforms as transforms
from torchvision import models
from ultralytics import YOLO
import numpy as np
from collections import OrderedDict, Counter
import json
import random
import base64
import io
from PIL import Image
import tempfile
import os
from typing import List, Tuple, Dict, Any
import logging
from contextlib import asynccontextmanager
from gtts import gTTS
import uuid
import asyncio
from concurrent.futures import ThreadPoolExecutor
import requests
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for models
yolo_model = None
scene_classifier = None
nlg = None

# ================== Model Download Helper ==================
def download_file(url: str, filename: str):
    """Download file from URL if it doesn't exist locally"""
    if os.path.exists(filename):
        logger.info(f"✓ {filename} already exists")
        return True
    
    try:
        logger.info(f"Downloading {filename}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(filename, 'wb') as f:
            shutil.copyfileobj(response.raw, f)
        
        logger.info(f"✓ {filename} downloaded successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to download {filename}: {e}")
        return False

# ================== Scene Classification Module ==================
class SceneClassifier:
    def __init__(self, model_path='scene.pth.tar', categories_file='categories_places365.txt'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Load classes
        self.classes = []
        try:
            if os.path.exists(categories_file):
                with open(categories_file) as class_file:
                    for line in class_file:
                        self.classes.append(line.strip().split(' ')[0][3:])
            else:
                logger.warning(f"Categories file {categories_file} not found, using default classes")
                # Default scene classes if file not found
                self.classes = ['indoor', 'outdoor', 'street', 'kitchen', 'bedroom', 'living_room', 
                              'bathroom', 'office', 'park', 'beach', 'restaurant', 'shop', 'car', 'train']
        except Exception as e:
            logger.error(f"Error loading categories: {e}")
            self.classes = ['indoor', 'outdoor', 'street', 'kitchen', 'bedroom', 'living_room']
        
        self.classes = tuple(self.classes)
        logger.info(f"Loaded {len(self.classes)} scene categories")
        
        # Load model
        self.model = models.resnet50(num_classes=len(self.classes))
        try:
            if os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location=self.device)
                state_dict = checkpoint['state_dict']
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    new_key = k.replace('module.', '')
                    new_state_dict[new_key] = v
                self.model.load_state_dict(new_state_dict)
                logger.info("Scene classification model loaded successfully")
            else:
                logger.warning(f"Scene model {model_path} not found, using pretrained ResNet50")
                # Use pretrained model if custom model not available
                self.model = models.resnet50(pretrained=True)
        except Exception as e:
            logger.error(f"Error loading scene model: {e}")
            self.model = models.resnet50(pretrained=True)
        
        self.model.to(self.device)
        self.model.eval()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def predict(self, frame):
        try:
            # Convert frame to PIL Image
            if isinstance(frame, np.ndarray):
                if len(frame.shape) == 3:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(frame_rgb)
                else:
                    pil_image = Image.fromarray(frame)
            else:
                pil_image = frame
            
            input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                top_prob, top_class = torch.topk(probabilities, 1)
                
            scene_class = self.classes[top_class.item()] if top_class.item() < len(self.classes) else "unknown"
            confidence = top_prob.item()
            
            return scene_class, confidence
        except Exception as e:
            logger.error(f"Scene classification error: {e}")
            return "unknown", 0.0

# ================== Natural Language Generation Module ==================
class NaturalLanguageGenerator:
    def __init__(self, confidence_threshold=0.3):
        self.confidence_threshold = confidence_threshold
        
        self.templates = {
            'basic': [
                "I can see {objects} in this {scene}.",
                "This {scene} contains {objects}.",
                "In this {scene} setting, there are {objects}.",
                "The {scene} shows {objects}."
            ],
            'descriptive': [
                "This {scene} scene features {objects}.",
                "Looking at this {scene} view, I observe {objects}.",
                "The {scene} environment displays {objects}.",
                "This {scene} area has {objects}."
            ],
            'no_objects': [
                "This is a {scene} scene with no prominent objects visible.",
                "The {scene} appears clear.",
                "This {scene} setting shows no notable items at the moment."
            ],
            'movement': [
                "Currently in a {scene}, I can see {objects}.",
                "Moving through a {scene} area with {objects}.",
                "The camera shows a {scene} with {objects}."
            ]
        }
    
    def format_objects(self, objects):
        if not objects:
            return ""
        
        # Count objects
        object_counts = Counter(objects)
        formatted = []
        
        for obj, count in object_counts.items():
            if count == 1:
                formatted.append(f"a {obj}")
            else:
                formatted.append(f"{count} {obj}s")
        
        if len(formatted) == 1:
            return formatted[0]
        elif len(formatted) == 2:
            return f"{formatted[0]} and {formatted[1]}"
        else:
            return ", ".join(formatted[:-1]) + f", and {formatted[-1]}"
    
    def generate_description(self, detected_objects, scene_label, include_movement=False):
        # Filter objects by confidence
        filtered_objects = [
            obj_name for obj_name, confidence in detected_objects
            if confidence >= self.confidence_threshold
        ]
        
        # Clean scene label
        scene = scene_label.lower().replace('_', ' ')
        
        if filtered_objects:
            object_string = self.format_objects(filtered_objects[:5])  # Limit to 5 objects
            if include_movement:
                template = random.choice(self.templates['movement'])
            else:
                template = random.choice(self.templates['basic'] + self.templates['descriptive'])
            description = template.format(scene=scene, objects=object_string)
        else:
            template = random.choice(self.templates['no_objects'])
            description = template.format(scene=scene)
        
        return description.capitalize()

# ================== Model Loading ==================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load models on startup
    global yolo_model, scene_classifier, nlg
    
    logger.info("Starting Video Narrator API...")
    logger.info("Loading models...")
    
    try:
        # Create temp directory for models if needed
        os.makedirs('/tmp/models', exist_ok=True)
        
        # Load YOLO model
        yolo_path = 'best.pt'
        if os.path.exists(yolo_path):
            logger.info("Loading custom YOLO model...")
            yolo_model = YOLO(yolo_path)
            logger.info("✓ Custom YOLO model loaded")
        else:
            logger.warning("Custom YOLO model not found, using YOLOv8n")
            yolo_model = YOLO('yolov8n.pt')  # Will download automatically
            logger.info("✓ Default YOLO model loaded")
        
        # Load Scene Classifier
        logger.info("Loading Scene Classifier...")
        scene_classifier = SceneClassifier()
        logger.info("✓ Scene classifier loaded")
        
        # Load NLG
        nlg = NaturalLanguageGenerator()
        logger.info("✓ NLG module loaded")
        
        logger.info("🚀 All models loaded successfully!")
        logger.info("API is ready for mobile app integration!")
        
    except Exception as e:
        logger.error(f"❌ Error loading models: {e}")
        # Don't raise - let app start with basic functionality
        logger.info("Starting with basic functionality...")
    
    yield
    
    # Cleanup on shutdown
    logger.info("Shutting down Video Narrator API...")

# ================== FastAPI App ==================
app = FastAPI(
    title="Video Narrator API",
    description="AI-powered video narration API for blind assistance - Mobile Ready",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware - Essential for mobile apps
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your mobile app domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Thread pool for CPU-intensive tasks
executor = ThreadPoolExecutor(max_workers=2)  # Reduced for free tier

# ================== API Endpoints ==================

@app.get("/")
async def root():
    return {
        "message": "Video Narrator API for Blind Assistance - Mobile Ready",
        "version": "1.0.0",
        "status": "active",
        "mobile_endpoints": [
            "POST /analyze-frame - Send camera frame for analysis",
            "POST /analyze-frame-tts - Get analysis + audio",
            "POST /batch-analyze - Process multiple frames",
            "GET /health - Check API status",
            "GET /keep-alive - Prevent cold starts"
        ],
        "deployment": "Render",
        "optimized_for": "Mobile Applications"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": asyncio.get_event_loop().time(),
        "models_loaded": {
            "yolo": yolo_model is not None,
            "scene_classifier": scene_classifier is not None,
            "nlg": nlg is not None
        },
        "device": str(scene_classifier.device) if scene_classifier else "cpu",
        "memory_usage": "optimized_for_mobile",
        "ready_for_mobile": True
    }

@app.get("/keep-alive")
async def keep_alive():
    """Endpoint for mobile apps to prevent cold starts"""
    return {
        "status": "alive",
        "timestamp": asyncio.get_event_loop().time(),
        "message": "API is warm and ready"
    }

def process_frame_sync(image_data: bytes):
    """Synchronous frame processing function - Optimized for mobile"""
    try:
        # Decode image
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise ValueError("Invalid image data")
        
        # Resize frame for faster processing on mobile
        height, width = frame.shape[:2]
        if width > 640:
            scale = 640 / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            frame = cv2.resize(frame, (new_width, new_height))
        
        # Object detection
        detected_objects = []
        try:
            results = yolo_model(frame, verbose=False, conf=0.4)  # Higher confidence for mobile
            
            if results and len(results) > 0:
                result = results[0]
                if hasattr(result, 'boxes') and result.boxes is not None:
                    for box in result.boxes:
                        if hasattr(box, 'cls') and hasattr(box, 'conf'):
                            class_id = int(box.cls.cpu().numpy())
                            confidence = float(box.conf.cpu().numpy())
                            
                            if hasattr(result, 'names'):
                                class_name = result.names[class_id]
                                detected_objects.append((class_name, confidence))
        except Exception as e:
            logger.error(f"Object detection error: {e}")
        
        # Scene classification
        scene_label = "unknown"
        scene_confidence = 0.0
        try:
            scene_label, scene_confidence = scene_classifier.predict(frame)
        except Exception as e:
            logger.error(f"Scene classification error: {e}")
        
        # Generate description
        description = "Scene analysis in progress..."
        try:
            if detected_objects or scene_label != "unknown":
                description = nlg.generate_description(detected_objects, scene_label, include_movement=True)
        except Exception as e:
            logger.error(f"Description generation error: {e}")
        
        return {
            "description": description,
            "objects": [{"name": obj[0], "confidence": round(obj[1], 2)} for obj in detected_objects[:10]],
            "scene": {
                "label": scene_label,
                "confidence": round(scene_confidence, 2)
            },
            "object_count": len(detected_objects),
            "processing_time": "optimized_for_mobile",
            "frame_processed": True
        }
        
    except Exception as e:
        logger.error(f"Frame processing error: {e}")
        raise HTTPException(status_code=500, detail=f"Frame processing failed: {str(e)}")

@app.post("/analyze-frame")
async def analyze_frame(file: UploadFile = File(...)):
    """Analyze a single frame and return description - Mobile Optimized"""
    
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image data
        image_data = await file.read()
        
        # Validate file size (mobile optimization)
        if len(image_data) > 5 * 1024 * 1024:  # 5MB limit
            raise HTTPException(status_code=400, detail="Image too large. Max 5MB for mobile processing.")
        
        # Process frame in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(executor, process_frame_sync, image_data)
        
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"Error in analyze_frame: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-frame-tts")
async def analyze_frame_with_tts(file: UploadFile = File(...)):
    """Analyze frame and return both description and audio file - Mobile Ready"""
    
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Get analysis result
        image_data = await file.read()
        
        # Validate file size
        if len(image_data) > 5 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="Image too large. Max 5MB for mobile processing.")
        
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(executor, process_frame_sync, image_data)
        
        # Generate TTS audio
        try:
            tts = gTTS(text=result["description"], lang='en', slow=False)
            
            # Save to temporary file
            audio_filename = f"narration_{uuid.uuid4().hex[:8]}.mp3"
            audio_path = f"/tmp/{audio_filename}"
            tts.save(audio_path)
            
            # Add audio info to result
            result["audio_file"] = audio_filename
            result["audio_url"] = f"/audio/{audio_filename}"
            result["has_audio"] = True
            
        except Exception as e:
            logger.error(f"TTS generation error: {e}")
            result["audio_error"] = "TTS generation failed"
            result["has_audio"] = False
        
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"Error in analyze_frame_with_tts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/audio/{filename}")
async def get_audio_file(filename: str):
    """Serve generated audio files"""
    audio_path = f"/tmp/{filename}"
    
    if os.path.exists(audio_path):
        return FileResponse(
            audio_path,
            media_type="audio/mpeg",
            filename=filename,
            headers={"Cache-Control": "public, max-age=3600"}  # Cache for 1 hour
        )
    else:
        raise HTTPException(status_code=404, detail="Audio file not found")

@app.post("/batch-analyze")
async def batch_analyze_frames(files: List[UploadFile] = File(...)):
    """Analyze multiple frames - Optimized for mobile video processing"""
    
    if len(files) > 5:  # Reduced batch size for mobile
        raise HTTPException(status_code=400, detail="Maximum 5 files per batch for mobile optimization")
    
    results = []
    
    for file in files:
        if not file.content_type.startswith('image/'):
            continue
            
        try:
            image_data = await file.read()
            
            # Skip large files
            if len(image_data) > 5 * 1024 * 1024:
                results.append({"error": "File too large", "filename": file.filename})
                continue
                
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(executor, process_frame_sync, image_data)
            result["filename"] = file.filename
            results.append(result)
        except Exception as e:
            logger.error(f"Error processing file {file.filename}: {e}")
            results.append({"error": str(e), "filename": file.filename})
    
    return JSONResponse(content={
        "results": results, 
        "processed_count": len(results),
        "mobile_optimized": True
    })

@app.get("/mobile-info")
async def mobile_integration_info():
    """Information for mobile app developers"""
    return {
        "integration_guide": {
            "camera_frames": "Send frames to POST /analyze-frame",
            "live_narration": "Use POST /analyze-frame-tts for audio",
            "batch_processing": "Use POST /batch-analyze for multiple frames",
            "keep_alive": "Ping GET /keep-alive to prevent cold starts"
        },
        "recommendations": {
            "frame_rate": "Send 1 frame every 2-3 seconds for optimal performance",
            "image_size": "Resize images to max 640px width before sending",
            "file_size": "Keep images under 5MB",
            "error_handling": "Always handle network timeouts gracefully"
        },
        "android_example": {
            "http_client": "Use Retrofit or OkHttp",
            "multipart": "Send images as multipart/form-data",
            "audio_playback": "Use MediaPlayer for returned audio URLs"
        }
    }

# ================== Error Handlers ==================
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code, "mobile_friendly": True}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "status_code": 500, "mobile_friendly": True}
    )

# ================== Main Entry Point ==================
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    host = "0.0.0.0"  # Required for Render deployment
    
    logger.info(f"Starting Video Narrator API on {host}:{port}")
    logger.info("Ready for mobile app integration!")
    
    uvicorn.run(
        app, 
        host=host, 
        port=port, 
        log_level="info",
        access_log=True
    )