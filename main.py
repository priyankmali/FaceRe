from fastapi import FastAPI, File, UploadFile, Form
import numpy as np
import cv2
import face_recognition
from fastapi.middleware.cors import CORSMiddleware
import json
from fastapi.responses import JSONResponse


app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/encode")
async def encode_face(file: UploadFile = File(...)):

    image_bytes = await file.read()
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Detect & encode face
    face_locations = face_recognition.face_locations(img)
    if len(face_locations) == 0:
        return {"error": "No face detected"}
    
    encodings = face_recognition.face_encodings(img, face_locations)

    if len(encodings) == 0:
        return {"error": "Encoding failed"}

    # Convert numpy array -> list for JSON
    encoding_list = encodings[0].tolist()

    return {"encoding": encoding_list}




@app.post("/recognize/")
async def recognize(face_image: UploadFile = File(...), known_faces: str = Form(...)):
    # Parse known faces

    known_faces = json.loads(known_faces)
    valid_faces = [face for face in known_faces if face["encoding"] is not None]
    known_encodings = [np.array(face["encoding"], dtype="float32").flatten() for face in valid_faces]
    known_users = [{"employee_id": face["employee_id"], "name": face["name"]} for face in valid_faces]

    # read uploaded image
    image_bytes = await face_image.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)


    # Detect faces in frame
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    if not face_encodings:
        return JSONResponse({'status': 'no_face','message': 'No face detected in the image'})
    
    results = []
    # compare with known faces
    for encoding in face_encodings:
        matches = face_recognition.compare_faces(known_encodings, encoding, tolerance=0.38)
        face_distances = face_recognition.face_distance(known_encodings, encoding)
        
        best_match_index = np.argmin(face_distances) if len(face_distances) > 0 else None
        
        if best_match_index is not None and matches[best_match_index]:
            user = known_users[best_match_index]
            
            results.append({
                "user": user,
                "name": user["name"],
                "distance": float(face_distances[best_match_index])})
        else:
            results.append({"user": None, "status": "unknown"})
            return JSONResponse({'status': 'no_face','message': 'No face detected in the image'})
    return {"status": "success", "results": results}



    
