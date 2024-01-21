import os
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import np_utils
import pygame

# Define constants
threshold = 25
alarm_sound_path = "alarm.wav"

# Load the training data
train_data = []
train_labels = []

for label in ["Closed_Eyes", "Open_Eyes"]:
    folder_path = "train/" + label
    for img_path in os.listdir(folder_path):
        img = cv2.imread(os.path.join(folder_path, img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (24, 24))
        train_data.append(img)
        if label == "Open_Eyes":
            train_labels.append(1)
        else:
            train_labels.append(0)

train_data = np.array(train_data)
train_labels = np.array(train_labels)

# Convert labels to one-hot encoding
num_classes = 2
train_labels = np_utils.to_categorical(train_labels, num_classes)

# Define the model architecture
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(24, 24, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(train_data.reshape(-1, 24, 24, 1), train_labels, batch_size=32, epochs=10, verbose=1)

# Initialize the counter and the score
score = 0

# Start the video stream
cap = cv2.VideoCapture(0)

while True:
    # Read the frame from the video stream
    ret, frame = cap.read()

    if not ret:
        break

    # Convert the frame to grayscale and equalize the histogram
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    # Detect the face in the image
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Loop over all the detected faces
    for (x,y,w,h) in faces:
        # Extract the region of interest (ROI) containing the face
        roi_gray = gray[y:y+h, x:x+w]

        # Detect the eyes in the ROI
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        eyes = eye_cascade.detectMultiScale(roi_gray)

        # Loop over all the detected eyes
        for (ex,ey,ew,eh) in eyes:
            # Extract the region of interest (ROI) containing the eye
            roi_eye = roi_gray[ey:ey+eh, ex:ex+ew]

            # Preprocess the image of the eye
            roi_eye = cv2.resize(roi_eye, (24, 24))
            roi_eye = roi_eye.astype('float32') / 255.0
            # Reshape the image to match the input shape of the model
            roi_eye = roi_eye.reshape(-1, 24, 24, 1)

            # Make a prediction using the model
            prediction = model.predict(roi_eye)

            # Check if the eye is closed
            if np.argmax(prediction) == 0:
                label = "Closed"
                score += 1
                if score > 50:  # Reduced threshold for more sensitivity
                    pygame.mixer.init()
                    pygame.mixer.music.load(alarm_sound_path)
                    pygame.mixer.music.play()
                    cv2.putText(frame, "ALERT: SLEEP DETECTED!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                label = "Open"
                score = 0

            # Draw a rectangle around the eye and label it
            cv2.rectangle(frame, (x+ex,y+ey), (x+ex+ew,y+ey+eh), (0,255,0), 2)
            cv2.putText(frame, label, (x+ex, y+ey-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('frame',frame)

    # Check if the user wants to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
