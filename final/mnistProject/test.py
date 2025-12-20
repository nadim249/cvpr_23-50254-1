import cv2
import numpy as np
import tensorflow as tf

# Setup
model = tf.keras.models.load_model("digit_recognition_model.keras")
cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    
    # 1. CROP CENTER: 300x300 box in the middle
    h, w = frame.shape[:2]
    y, x = (h - 300) // 2, (w - 300) // 2
    roi = frame[y:y+300, x:x+300]

    # 2. PREPARE IMAGE:
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    processed = cv2.resize(thresh, (28, 28)).reshape(1, 784) / 255.0

    # 3. PREDICT
    digit = np.argmax(model.predict(processed, verbose=0))

    # Draw box and result
    cv2.rectangle(frame, (x, y), (x+300, y+300), (0, 255, 0), 2)
    cv2.putText(frame, f"Digit: {digit}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow("Digit Recognition", frame)
    if cv2.waitKey(1) == ord('q'): break

cap.release()
cv2.destroyAllWindows()