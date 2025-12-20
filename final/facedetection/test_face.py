import cv2
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("face_attendance_model1.keras")

class_names = ['bikash', 'marowa', 'mehedi', 'nadim', 'unknown']
KNOWN_CLASSES = class_names[:-1]
IMG_SIZE = 256
CONFIDENCE_THRESHOLD = 0.70

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
        face = face / 255.0
        face = np.expand_dims(face, axis=0)

        preds = model.predict(face, verbose=0)
        class_id = np.argmax(preds)
        confidence = np.max(preds)

        name = class_names[class_id]

        # Decision logic
        if name != "unknown" and confidence >= CONFIDENCE_THRESHOLD:
            color = (0, 255, 0)  # GREEN
            label = f"{name} ({confidence:.2f})"
        else:
            color = (0, 0, 255)  # RED
            label = "Unknown"

        # Draw
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Face Attendance System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
