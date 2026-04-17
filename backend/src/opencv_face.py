import cv2

class OpenCVFaceDetector:
    def __init__(self):
        # Use OpenCV's built-in face detection
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            print("Warning: Could not load face cascade classifier")

    def get_bbox(self, img):
        if img is None:
            return None
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60)
        )
        
        if len(faces) == 0:
            return None
            
        # Return the first face
        x, y, w, h = faces[0]
        return [int(x), int(y), int(w), int(h)]
