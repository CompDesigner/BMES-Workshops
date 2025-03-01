import cv2 as cv 
import mediapipe as mp
import numpy as npy
import hashlib
import tensorflow as tf
#from deepface import DeepFace 

face = mp.solutions.face_mesh
face_mesh = face.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

#Load Age and Gender models
age_net = cv.dnn.readNetFromCaffe('deploy_age.prototxt', 'age_net.caffemodel')
gender_net = cv.dnn.readNetFromCaffe('deploy_gender.prototxt', 'gender_net.caffemodel')

path = cv.VideoCapture(0) 

#Classification for age and gender
age_list = ['(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
gender_list = ['Male', 'Female']

"""
age_buffer = []
gender_buffer = [] """

#Generating Bio-ID
def generate_bio_id(landmark):
    landmark_flat = npy.array(landmark).flatten()
    bio_id = hashlib.sha256(landmark_flat.tobytes()).hexdigest()[:16]
    return bio_id

"""
def crop(frame,  landmarks_px):
    x_min = max(0, min([x for x, y in landmarks_px]))
    y_min = max(0, min([y for x, y in landmarks_px]))
    x_max = min(frame.shape[1], max([x for x, y in landmarks_px]))
    y_max = min(frame.shape[0], max([y for x, y in landmarks_px]))
    return frame[y_min:y_max, x_min:x_max]
"""

while True:
    passes, frame = path.read()
    if not passes:
        break

    #Convert frame to RGB for Mediapipe
    rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    #Process the frame to detect face landmarks
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            #Collect landmark coordinates
            landmarks = [(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark]
            
            #Convert normalized coordinates to pixel coordinates
            h, w, _ = frame.shape
            landmarks_px = [(int(x * w), int(y * h)) for x, y, z in landmarks]

            #cropped = crop(frame, landmarks_px)

            #Draw landmarks on the face
            for (x, y) in landmarks_px:
                cv.circle(frame, (x, y), 1, (0, 255, 0), -1)

            #Generate and display Bio-ID
            bio_id = generate_bio_id(landmarks)
            cv.putText(frame, f"Bio-ID: {bio_id}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

            
            # Highlight key facial features (e.g., eyes, nose, mouth)
            eye_left = landmarks_px[33:133]  # Approx. left eye region
            eye_right = landmarks_px[362:462]  # Approx. right eye region
            nose = landmarks_px[1:5]  # Nose bridge region
            mouth = landmarks_px[78:308]  # Mouth region

            #Draw polylines around key features
            cv.polylines(frame, [npy.array(eye_left)], isClosed=True, color=(0, 255, 255), thickness=1)
            cv.polylines(frame, [npy.array(eye_right)], isClosed=True, color=(0, 255, 255), thickness=1)
            cv.polylines(frame, [npy.array(nose)], isClosed=False, color=(255, 0, 255), thickness=1)
            cv.polylines(frame, [npy.array(mouth)], isClosed=True, color=(0, 255, 0), thickness=1)
            

            #Preprocess the face for age and gender classification
            #if cropped.shape[0] > 0 and cropped.shape[1] > 0:
            """
            #Use DeepFace for age and gender analysis
                try:
                    analysis = DeepFace.analyze(face_crop, actions=['age', 'gender'], enforce_detection=False)

                    # Extract age and gender predictions
                    age = analysis['age']
                    gender = analysis['gender']
            """
            face_blob = cv.dnn.blobFromImage(frame, 1.0, (227, 227), (104, 117, 123), swapRB=False, crop=False)
            age_net.setInput(face_blob)
            gender_net.setInput(face_blob)

            #Predict age and gender
            age_preds = age_net.forward()
            gender_preds = gender_net.forward()

            
            age = age_list[npy.argmax(age_preds)]
            gender = gender_list[npy.argmax(gender_preds)]
            """
                # Filter based on confidence
                age_confidence = npy.max(age_preds)
                gender_confidence = npy.max(gender_preds)
                age = age_list[npy.argmax(age_preds)] if age_confidence > 0.6 else "Uncertain"
                gender = gender_list[npy.argmax(gender_preds)] if gender_confidence > 0.6 else "Uncertain"

                # Add predictions to buffers
                age_buffer.append(age)
                gender_buffer.append(gender)

                # Smooth predictions using a moving average
                if len(age_buffer) > 10:  # Keep buffer size to last 10 frames
                    age_buffer.pop(0)
                    gender_buffer.pop(0)

                smoothed_age = max(set(age_buffer), key=age_buffer.count)
                smoothed_gender = max(set(gender_buffer), key=gender_buffer.count)
            """
            cv.putText(frame, f"Age: {age}", (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv.putText(frame, f"Gender: {gender}", (10, 90), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            """
            except Exception as e:
                    print(f"DeepFace analysis error: {e}")
            """

    cv.imshow("Facial Feature Detection with Bio-ID", frame)

    if cv.waitKey(5) & 0xFF == 27:
        break

path.release()
cv.destroyAllWindows()
