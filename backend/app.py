"""
KageN AI Anti-Spoofing Detector Backend
REST API for advanced liveness and spoof detection
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
import cv2
import numpy as np
import base64
import os
import sys

# Use absolute paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, 'src'))

# Now import from src
from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name

# Use absolute paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "resources/anti_spoof_models")
DETECTION_MODEL_DIR = os.path.join(SCRIPT_DIR, "resources/detection_model")

app = FastAPI(title="KageN AI Anti-Spoofing Detector API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("� Starting KageN AI Anti-Spoofing Detector...")
print(f"   Backend root: {SCRIPT_DIR}")
print(f"   Model weights: {MODEL_DIR}")
print(f"   Face detection config: {DETECTION_MODEL_DIR}")

try:
    model_test = AntiSpoofPredict(0)  # 0 = CPU mode
    image_cropper = CropImage()
    print("✅ KageN AI models loaded and ready!")
except Exception as e:
    print(f"❌ [KageN AI] Model initialization failed: {e}")
    model_test = None
    image_cropper = None


# ========== ROUTES ==========

@app.get("/")
async def root():
    """KageN AI: Camera Liveness Detection Demo"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=no">
    <title>KageN AI Liveness Detection</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { 
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
                color: #fff;
                height: 100vh;
                overflow: hidden;
            }
            
            #video {
                width: 100%;
                height: 100vh;
                object-fit: cover;
                position: absolute;
                top: 0;
                left: 0;
            }
            
            canvas { display: none; }
            
            #overlay {
                position: fixed;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                width: 320px;
                height: 320px;
                border-radius: 160px;
                border: 4px solid #00ff00;
                box-shadow: 0 0 0 9999px rgba(0, 0, 0, 0.5);
                pointer-events: none;
                z-index: 10;
            }
            
            #status-container {
                position: fixed;
                bottom: 50px;
                left: 50%;
                transform: translateX(-50%);
                width: 90%;
                max-width: 500px;
                background: rgba(0, 0, 0, 0.9);
                border-radius: 20px;
                padding: 25px;
                text-align: center;
                backdrop-filter: blur(10px);
                border: 2px solid rgba(255, 255, 255, 0.2);
                z-index: 20;
            }
            
            #status-text {
                font-size: 26px;
                font-weight: bold;
                margin-bottom: 12px;
                text-shadow: 0 2px 4px rgba(0, 0, 0, 0.5);
            }
            
            #confidence-text {
                font-size: 18px;
                opacity: 0.9;
                margin-bottom: 10px;
            }
            
            #analysis-text {
                font-size: 14px;
                opacity: 0.7;
                font-family: 'Courier New', monospace;
            }
            
            .real { 
                color: #00ff00;
                text-shadow: 0 0 10px #00ff00;
            }
            
            .fake { 
                color: #ff4444;
                text-shadow: 0 0 10px #ff4444;
            }
            
            .analyzing { 
                color: #ffaa00;
                animation: pulse 1.5s infinite;
            }
            
            @keyframes pulse {
                0%, 100% { opacity: 1; }
                50% { opacity: 0.6; }
            }
            
            #header {
                position: fixed;
                top: 20px;
                left: 20px;
                background: rgba(0, 0, 0, 0.7);
                border-radius: 15px;
                padding: 12px 20px;
                font-size: 14px;
                backdrop-filter: blur(10px);
                z-index: 20;
            }
            
            #controls {
                position: fixed;
                top: 20px;
                right: 20px;
                z-index: 20;
            }
            
            button {
                background: rgba(0, 0, 0, 0.8);
                color: white;
                border: 2px solid rgba(255, 255, 255, 0.3);
                border-radius: 25px;
                padding: 12px 24px;
                font-size: 16px;
                font-weight: bold;
                backdrop-filter: blur(10px);
                cursor: pointer;
                transition: all 0.3s ease;
            }
            
            button:hover {
                background: rgba(0, 0, 0, 0.95);
                border-color: rgba(255, 255, 255, 0.5);
            }
            
            button:active {
                transform: scale(0.95);
            }
            
            #frame-counter {
                position: fixed;
                bottom: 20px;
                right: 20px;
                background: rgba(0, 0, 0, 0.7);
                border-radius: 10px;
                padding: 8px 15px;
                font-size: 12px;
                backdrop-filter: blur(10px);
                z-index: 20;
            }
        </style>
    </head>
    <body>
        <video id="video" autoplay playsinline></video>
        <canvas id="canvas"></canvas>
        <div id="overlay"></div>
        
        <div id="header">
            🎥 KageN AI Liveness Detection
        </div>
        
        <div id="controls">
            <button id="toggleBtn">⏸️ Pause</button>
        </div>
        
        <div id="status-container">
            <div id="status-text" class="analyzing">📷 Initializing...</div>
            <div id="confidence-text"></div>
            <div id="analysis-text"></div>
        </div>
        
        <div id="frame-counter">Frames: 0</div>

        <script>
            const video = document.getElementById('video');
            const canvas = document.getElementById('canvas');
            const ctx = canvas.getContext('2d');
            const statusText = document.getElementById('status-text');
            const confidenceText = document.getElementById('confidence-text');
            const analysisText = document.getElementById('analysis-text');
            const toggleBtn = document.getElementById('toggleBtn');
            const frameCounter = document.getElementById('frame-counter');
            
            let isRunning = true;
            let isDetecting = false;
            let frameCount = 0;
            
            // Initialize camera
            async function initCamera() {
                try {
                    const stream = await navigator.mediaDevices.getUserMedia({
                        video: {
                            facingMode: 'user',
                            width: { ideal: 640 },
                            height: { ideal: 480 }
                        }
                    });
                    video.srcObject = stream;
                    statusText.textContent = '👤 Position your face in the circle';
                    statusText.className = '';
                    startDetection();
                } catch (error) {
                    statusText.textContent = '❌ Camera access denied';
                    statusText.className = 'fake';
                    console.error('Camera error:', error);
                }
            }
            
            // Capture and detect
            async function captureAndDetect() {
                if (!isRunning || isDetecting) return;
                
                isDetecting = true;
                frameCount++;
                frameCounter.textContent = `Frames: ${frameCount}`;
                
                try {
                    // Set canvas size to match video
                    canvas.width = video.videoWidth;
                    canvas.height = video.videoHeight;
                    
                    // Draw video frame
                    ctx.drawImage(video, 0, 0);
                    
                    // Get base64
                    const base64Image = canvas.toDataURL('image/jpeg', 0.85).split(',')[1];
                    
                    // Update UI - analyzing
                    statusText.textContent = '🔍 Analyzing...';
                    statusText.className = 'analyzing';
                    confidenceText.textContent = '';
                    analysisText.textContent = '';
                    
                    // Call API
                    const response = await fetch('/detect_liveness', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ image: base64Image })
                    });
                    
                    const result = await response.json();
                    
                    // Update UI based on result
                    if (!result.face_detected) {
                        statusText.textContent = '👤 No face detected';
                        statusText.className = '';
                        confidenceText.textContent = 'Position your face clearly in the circle';
                        analysisText.textContent = '';
                    } else if (result.is_real) {
                        statusText.textContent = '✅ REAL PERSON';
                        statusText.className = 'real';
                        confidenceText.textContent = `Confidence: ${(result.confidence * 100).toFixed(1)}%`;
                        analysisText.textContent = result.analysis || 'Live person detected!';
                    } else {
                        statusText.textContent = '❌ SPOOF DETECTED';
                        statusText.className = 'fake';
                        confidenceText.textContent = `Spoof Score: ${(result.confidence * 100).toFixed(1)}%`;
                        analysisText.textContent = result.analysis || 'Photo or video detected!';
                    }
                    
                } catch (error) {
                    console.error('Detection error:', error);
                    statusText.textContent = '⚠️ Network error';
                    statusText.className = '';
                } finally {
                    isDetecting = false;
                }
            }
            
            // Start detection loop (every 800ms for stability)
            function startDetection() {
                setInterval(() => {
                    captureAndDetect();
                }, 800);
            }
            
            // Toggle pause/resume
            toggleBtn.addEventListener('click', () => {
                isRunning = !isRunning;
                toggleBtn.textContent = isRunning ? '⏸️ Pause' : '▶️ Resume';
                
                if (!isRunning) {
                    statusText.textContent = '⏸️ Paused';
                    statusText.className = '';
                    confidenceText.textContent = 'Tap Resume to continue';
                    analysisText.textContent = '';
                } else {
                    statusText.textContent = '👤 Position your face in the circle';
                    statusText.className = '';
                    confidenceText.textContent = '';
                    analysisText.textContent = '';
                }
            });
            
            // Start
            initCamera();
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model_test is not None,
        "service": "KageN AI Anti-Spoofing Detector API"
    }


@app.post("/detect_liveness")
async def detect_liveness(data: dict):
    """
    Liveness detection endpoint using KageN AI models
    
    Request:
        {
            "image": "base64_encoded_image"
        }
    
    Response:
        {
            "is_real": bool,
            "confidence": float (0-1),
            "face_detected": bool,
            "label": str ("Real" or "Fake"),
            "analysis": str,
            "bbox": [x, y, w, h]
        }
    """
    try:
        if model_test is None:
            return JSONResponse({
                "is_real": False,
                "confidence": 0.0,
                "face_detected": False,
                "label": "Error",
                "analysis": "Model not loaded",
                "bbox": []
            })
        
        image_b64 = data.get("image", "")
        
        if not image_b64:
            return JSONResponse({
                "is_real": False,
                "confidence": 0.0,
                "face_detected": False,
                "label": "No Image",
                "analysis": "No image provided",
                "bbox": []
            })
        
        # Decode image
        try:
            image_bytes = base64.b64decode(image_b64)
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except Exception as e:
            print(f"Image decode error: {e}")
            return JSONResponse({
                "is_real": False,
                "confidence": 0.0,
                "face_detected": False,
                "label": "Decode Error",
                "analysis": "Could not decode image",
                "bbox": []
            })
        
        if image is None or image.size == 0:
            return JSONResponse({
                "is_real": False,
                "confidence": 0.0,
                "face_detected": False,
                "label": "Invalid Image",
                "analysis": "Could not process image",
                "bbox": []
            })
        
        # Get face bounding box
        try:
            image_bbox = model_test.get_bbox(image)
        except Exception as e:
            print(f"Face detection error: {e}")
            image_bbox = None
        
        if image_bbox is None:
            return JSONResponse({
                "is_real": False,
                "confidence": 0.0,
                "face_detected": False,
                "label": "No Face",
                "analysis": "No face detected in image",
                "bbox": []
            })
        
        # Run prediction with all models
        try:
            prediction = np.zeros((1, 3))
            
            for model_name in os.listdir(MODEL_DIR):
                if not model_name.endswith('.pth'):
                    continue
                
                h_input, w_input, model_type, scale = parse_model_name(model_name)
                
                params = {
                    "org_img": image,
                    "bbox": image_bbox,
                    "scale": scale,
                    "out_w": w_input,
                    "out_h": h_input,
                    "crop": True,
                }
                
                if scale is None:
                    params["crop"] = False
                
                try:
                    img = image_cropper.crop(**params)
                    model_prediction = model_test.predict(img, os.path.join(MODEL_DIR, model_name))
                    prediction += model_prediction
                except Exception as e:
                    print(f"Error in model {model_name}: {e}")
                    continue
            
            # Get label (1 = real, 0 = fake)
            label = int(np.argmax(prediction))
            confidence_score = float(prediction[0][label] / (np.sum(prediction) + 1e-6))
            
            # Determine if real or fake - convert numpy bool to Python bool
            is_real = bool(label == 1)  # 1 = Real, 0 = Fake
            
            return JSONResponse({
                "is_real": bool(is_real),
                "confidence": float(confidence_score),
                "face_detected": True,
                "label": "Real" if is_real else "Fake",
                "analysis": f"Model score: {float(prediction[0][label]):.4f} | Confidence: {float(confidence_score):.4f}",
                "bbox": [int(x) for x in image_bbox],
                "raw_prediction": [float(x) for x in prediction[0]]
            })
            
        except Exception as e:
            print(f"Prediction error: {e}")
            import traceback
            traceback.print_exc()
            
            return JSONResponse({
                "is_real": False,
                "confidence": 0.0,
                "face_detected": True,
                "label": "Error",
                "analysis": f"Prediction error: {str(e)}",
                "bbox": [int(x) for x in image_bbox]
            })
    
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        
        return JSONResponse({
            "is_real": False,
            "confidence": 0.0,
            "face_detected": False,
            "label": "Server Error",
            "analysis": "An unexpected error occurred",
            "bbox": []
        })


if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*60)
    print("🚀 KageN AI Anti-Spoofing Detector is live!")
    print("="*60)
    print("📍 Server running at: http://0.0.0.0:8000")
    print("🌐 Access at: http://localhost:8000")
    print("📊 API docs at: http://localhost:8000/docs")
    print("="*60 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
