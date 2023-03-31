import cv2
import math
import argparse
age_model_path = r"C:/gad/age_net.caffemodel"
age_config_path = r"C:/gad/age_deploy.prototxt"
gender_model_path = r"C:/gad/gender_net.caffemodel"
gender_config_path = r"C:/gad/gender_deploy.prototxt"

age_net = cv2.dnn.readNetFromCaffe(age_config_path, age_model_path)
gender_net = cv2.dnn.readNetFromCaffe(gender_config_path, gender_model_path)
img_path = "C:/gad/man2.jpg"
image = cv2.imread(img_path)
blob = cv2.dnn.blobFromImage(image, 1.0, (227, 227), [104, 117, 123], False, False)
gender_net.setInput(blob)
gender_preds = gender_net.forward()
gender_labels = ["Male", "Female"]
gender_idx = gender_preds[0].argmax()
gender = gender_labels[gender_idx]
age_net.setInput(blob)
age_preds = age_net.forward()
age = age_preds[0][0] * 100
(h, w) = image.shape[:2]
cv2.rectangle(image, (0, h - 70), (250, h), (0, 0, 0), -1)
cv2.putText(image, "Gender: {}".format(gender), (10, h - 50),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
cv2.putText(image, "Age: {:.1f}".format(age), (10, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

cv2.imshow("Output", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
