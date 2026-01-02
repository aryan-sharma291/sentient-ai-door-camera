
# TODO install this on raspberry pi python3 -m pip install tflite-runtime
import os
#conda activate fr312
# import tensorflow as tf
print("Working directory:", os.getcwd())
import face_recognition
image = face_recognition.load_image_file(
    "/Users/aryansharma/MySecondProject/SentientAI/NoteBooks/notebooks/picture1.jpeg"
)
face_locations = face_recognition.face_locations(image, model="cnn")


print(face_locations)
face_encodings = face_recognition.face_encodings(image, face_locations)
print(face_encodings)