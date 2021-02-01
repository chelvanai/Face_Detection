import imutils
import cv2
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import numpy as np
import csv
from face_detect.face_detector import FaceDetector


class VideoCamera:
    def __init__(self, video=None):
        self.input_file = str(video)
        print(self.input_file)
        self.cap = cv2.VideoCapture(video)

        self.fd = FaceDetector('face_detect/weight/model.pb')
        self.ec = load_model('emotion_detect/_mini_XCEPTION.102-0.66.hdf5', compile=False)

        self.emotions = ["angry", "disgust", "scared", "happy", "sad", "surprised",
                         "neutral"]

        self.file = open("emotions_results/" + self.input_file.split("/")[2].split('.')[0] + '.csv', 'w')
        self.csv_file = csv.writer(self.file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        self.csv_file.writerow(['Predicted Emotion', 'Probability'])

    def __del__(self):
        print("Destroyed")

    def get_frame(self):
        r, img = self.cap.read()

        if r is False:
            self.cap.release()
            cv2.destroyAllWindows()
            return None

        img = imutils.resize(img, width=1000)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces, scores = self.fd(img)

        print(faces)

        if len(faces) > 0:
            for face in faces:
                if len(faces) >1:
                    cv2.putText(img, 'Many faces detected', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (255, 255, 255), 2)

                ymin, xmin, ymax, xmax = int(face[0]), int(face[1]), int(face[2]), int(face[3])
                cropped_img = gray_img[ymin:ymax, xmin:xmax]

                roi = cv2.resize(cropped_img, (64, 64))
                roi = roi.astype("float") / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                preds = self.ec.predict(roi)[0]
                emotion_probability = np.max(preds)
                label = self.emotions[preds.argmax()]

                cv2.putText(img, str(label), (int(xmin) - 10, int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (255, 255, 255), 2)
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

                self.csv_file.writerow([label, emotion_probability])

        else:
            cv2.putText(img, 'No faces detected', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 255), 2)

        ret, jpeg = cv2.imencode('.jpg', img)
        return jpeg.tobytes()
