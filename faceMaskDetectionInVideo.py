import numpy as np
import cv2
from keras.models import load_model

# Load the face detection model
cvNet = cv2.dnn.readNetFromCaffe('MedicalMask/trainedFaceDetector/architecture.txt', 'MedicalMask/trainedFaceDetector/weights.caffemodel')

# Adjust the gamma of the image for better detection
def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)])
    return cv2.LUT(image.astype(np.uint8), table.astype(np.uint8))

img_size = 124

# Load the trained face mask detection model
trained_model = load_model('fmdTrainedModel/my_model.h5')

# Capture video from the webcam
video_capture = cv2.VideoCapture(0)  # 0 corresponds to the default camera

assign = {'0': 'Mask', '1': 'No Mask'}

while True:
    ret, frame = video_capture.read()

    gamma = 2.0
    frame = adjust_gamma(frame, gamma=gamma)
    (h, w) = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    cvNet.setInput(blob)
    detections = cvNet.forward()

    for i in range(0, detections.shape[2]):
        try:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            detected_face = frame[startY:endY, startX:endX]
            confidence = detections[0, 0, i, 2]

            if confidence > 0.2:
                im = cv2.resize(detected_face, (img_size, img_size))
                im = np.array(im) / 255.0
                im = im.reshape(1, img_size, img_size, 3)
                result = trained_model.predict(im)

                if result > 0.5:
                    label_Y = 1
                else:
                    label_Y = 0

                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
                cv2.putText(frame, assign[str(label_Y)], (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (36, 255, 12), 2)

        except:
            pass

    # Display the frame with face mask detection
    cv2.imshow('Video', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the OpenCV windows
video_capture.release()
cv2.destroyAllWindows()
