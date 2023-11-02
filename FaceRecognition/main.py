import cv2
import os
import face_recognition
import numpy as np
import math
import threading



def face_confidence(face_distance, face_match_threshold=0.6):
    range = (1.0 - face_match_threshold)
    linear_val = (1.0 - face_distance) / (range - 2.0)

    if face_distance > face_match_threshold:
        return str(round(linear_val * 100, 2)) + '%'
    else:
        value = (linear_val + ((1.0 - linear_val) * math.pow(abs(linear_val - 0.5) * 2, 0.2)))
        return str(round(value, 2)) + '%'

class FaceRecognition:
    face_location = []
    face_encodings = []
    face_names = []
    known_face_encodings = []
    known_face_names = []
    process_current_frame = True

    def __init__(self):
        self.encode_faces()

    def encode_faces(self):
        for image in os.listdir('faces'):
            face = face_recognition.load_image_file("faces/" + image)
            encoding = face_recognition.face_encodings(face)[0]

            self.known_face_encodings.append(encoding)
            self.known_face_names.append(image)

        print("faces List", self.known_face_names)
    def detect_faces(self, frame):
        if self.process_current_frame:
            self.face_locations = face_recognition.face_locations(frame)
            self.face_encodings = face_recognition.face_encodings(frame, self.face_locations,model='large')
            self.face_names = []
 # Set a default label color (e.g., red for unknown)
            label_color = (0, 0, 255)
            for face_encoding in self.face_encodings:
                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, 0.6)

                name = 'Unknown'
                confidence = 'Unknown'

                face_distances = face_recognition.face_distance(face_encoding,self.known_face_encodings)

                if True in matches:
                    best_match_index = matches.index(True)
                    name = self.known_face_names[best_match_index]
                    confidence = face_confidence(face_distances[best_match_index])
                    # Change label_color for known faces
                    label_color = (0, 255, 0)  # Green for recognized faces

                self.face_names.append(f'{name} ({confidence})')

            # Draw rectangles and labels on the frame for all detected faces
            for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
                cv2.rectangle(frame, (left, top), (right, bottom), label_color, 2)
                cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_DUPLEX, 0.5, label_color, 2)

            return frame

        self.process_current_frame = not self.process_current_frame
def video_capture_thread(fr):
    cap = cv2.VideoCapture(0)
    cap.set(3, 1024)  # Width
    cap.set(4, 980)  # Height
    while True:
        ret, frame = cap.read()
        frame = fr.detect_faces(frame)
        # frame = detect_faces_with_gpu(frame)
        if frame is not None:
            cv2.imshow('Video Face Detection', frame)

        if cv2.waitKey(1) & 0xff == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    fr = FaceRecognition()

    video_thread = threading.Thread(target=video_capture_thread, args=(fr,))
    video_thread.daemon = True
    video_thread.start()

    video_thread.join() 