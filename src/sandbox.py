from __future__ import print_function
import cv2 as cv
import argparse
import numpy as np
import base64
from PIL import Image
import tensorflow_hub as tfhub
import os
import matplotlib.pyplot as plt

class SSDMobileNetV2(object):
    FROZEN_GRAPH_PATH = "../res/frozen_inference_graph.pb"
    CONFIG_FILE_PATH = "../res/ssd_mobilenet_v2_coco_2018_03_29.pbtxt"
    LABELS_FILE = "../res/labels.txt"

    def __init__(self):
        self.class_labels = [] 

        #load model
        # self.detector = tfhub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/fpnlite_320x320/1")
        self.detector = cv.dnn_DetectionModel(self.FROZEN_GRAPH_PATH, self.CONFIG_FILE_PATH)
        # self.detector = cv.dnn.readNetFromTensorflow(self.FROZEN_GRAPH_PATH, self.CONFIG_FILE_PATH)
        self.detector.setInputSize(500,500)
        
        # load labels 
        with open(self.LABELS_FILE, "rt") as file:
            self.class_labels = file.read().rstrip("\n").split("\n")
        print("Initialized SSDMobileNetV2")

    def __call__(self, img : np.ndarray):
        ClassIndex, confidence , bbox = self.detector.detect(img, confThreshold=0.7)
        return self.draw_bboxes(img , ClassIndex, confidence , bbox )

    def draw_bboxes(self, img, class_index, confidences, bboxes):
        font_scale = 1
        font = cv.FONT_HERSHEY_PLAIN
        # output = (ClassIndex, confidence , bbox )
        if type(class_index) == np.ndarray :
            for index, conf, box in zip(class_index.flatten(), confidences.flatten(), bboxes):
                cv.rectangle(img, box, (255, 0, 0), 2)
                cv.putText(img, self.class_labels[index -1], (box[0] + 10, box[1] + 40), font, fontScale= font_scale, color=(0, 255, 0), thickness=3)
                cv.putText(img, str(conf), (box[0] + 10, box[1] + 40 + 40), font, fontScale= font_scale, color=(0, 255, 0), thickness=3)
        return img
# def draw_bboxes(img, class_labels, class_index, confidence_values, bboxes):
#         font_scale = 3
#         font = cv.FONT_HERSHEY_PLAIN
#         for index, conf, box in zip(class_index.flatten(), confidence_values.flatten(), bboxes):
#             cv.rectangle(img, box, (255, 0, 0), 2)
#             cv.putText(img, class_labels[index -1], (box[0] + 10, box[1] + 40), font, fontScale= font_scale, color=(0, 255, 0), thickness=3)

def rgba_2_rgb(img_rgba):
    rgb_img = Image.new("RGB", img_rgba.size, (255, 255, 255))
    rgb_img.paste(img_rgba, mask = img_rgba.split()[3])
    img = np.asarray(rgb_img)
    return img


def main():
    parser = argparse.ArgumentParser(description='Code for Cascade Classifier tutorial.')
    # parser.add_argument('--face_cascade', help='Path to face cascade.', default='../res/data/haarcascades/haarcascade_frontalface_alt.xml')
    parser.add_argument('--image', help='Path to image for detection.', default='../res/img_2.png')
    args = parser.parse_args()

    # face_cascade_name = args.face_cascade
    # detector = cv.CascadeClassifier()
    # #-- 1. Load the cascades
    # if not detector.load(cv.samples.findFile(face_cascade_name)):
    #     print('--(!)Error loading face cascade')
    #     exit(0)


    img_path = args.image
    img_load = Image.open(img_path)

    if img_load.mode == "RGBA":
        img = rgba_2_rgb(img_load)
    elif img_load.mode == "RGB":
        # Do nothing
        img =  np.asarray(img_load)
    else: 
        assert False, "unsupported format"
    # img_np = np.reshape(np.asarray(img_rgb), (1,img_rgb.shape[0], img_rgb.shape[1],3))

    print("img shape: ", img.shape)

    detector = SSDMobileNetV2()

    cam = cv.VideoCapture(0)
    if not cam.isOpened():
        print("cam couldnt connect")
        return 1

    while True:
        check, frame = cam.read()
        if not check:
            continue

        # frame = np.asarray(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
        img = detector(frame)
        

        if cv.waitKey(17) == 27:
            break
    
        cv.imshow("Video", frame)

    cam.release()
    cv.destroyAllWindows()

    # # different object detection models have additional results
    # # all of them are explained in the documentation
    # # result = {key:value.numpy() for key,value in results.items()}
    # print(results.keys())




        





    # img = base64.b64encode(img_np)
    # print(img)

    # while True:
    #     #-- 2. detect face
    #     processed_img = detect_faces(img, face_cascade)
    #     cv.imshow("display", img)
    #     if cv.waitKey(17) == 27:
    #         break

# def detect_faces(img, face_cascade):
#     img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#     img_gray = cv.equalizeHist(img_gray)
#     detected_faces= face_cascade.detectMultiScale(img_gray)
#     for (x,y,w,h) in detected_faces:
#         processed_img = cv.rectangle(img, (x,y), (x + w, y + h), (0,255,0), 3)
#     return processed_img

if __name__ == "__main__":
    main()



# def detectAndDisplay(frame):
#     frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
#     frame_gray = cv.equalizeHist(frame_gray)
#     #-- Detect faces
#     faces = face_cascade.detectMultiScale(frame_gray)
#     print(faces)
#     for (x,y,w,h) in faces:
#         center = (x + w//2, y + h//2)
#         # frame = cv.ellipse(frame, center, (w//2, h//2), 0, 0, 360, (255, 0, 255), 4)
#         frame = cv.rectangle(frame, (x,y), (x + w, y + h), (0,255,0), 3)
#         # faceROI = frame_gray[y:y+h,x:x+w]
#         # #-- In each face, detect eyes
#         # eyes = eyes_cascade.detectMultiScale(faceROI)
#         # for (x2,y2,w2,h2) in eyes:
#         #     eye_center = (x + x2 + w2//2, y + y2 + h2//2)
#         #     radius = int(round((w2 + h2)*0.25))
#         #     frame = cv.circle(frame, eye_center, radius, (255, 0, 0 ), 4)
        
#     cv.imshow('Capture - Face detection', frame)

# parser = argparse.ArgumentParser(description='Code for Cascade Classifier tutorial.')
# parser.add_argument('--face_cascade', help='Path to face cascade.', default='../res/data/haarcascades/haarcascade_frontalface_alt.xml')
# parser.add_argument('--camera', help='Camera divide number.', type=int, default=0)
# args = parser.parse_args()
# # parser.add_argument('--eyes_cascade', help='Path to eyes cascade.', default='../res/data/haarcascades/haarcascade_eye_tree_eyeglasses.xml')
# face_cascade_name = args.face_cascade
# # eyes_cascade_name = args.eyes_cascade
# face_cascade = cv.CascadeClassifier()
# # eyes_cascade = cv.CascadeClassifier()
# #-- 1. Load the cascades
# if not face_cascade.load(cv.samples.findFile(face_cascade_name)):
#     print('--(!)Error loading face cascade')
#     exit(0)
# # if not eyes_cascade.load(cv.samples.findFile(eyes_cascade_name)):
# #     print('--(!)Error loading eyes cascade')
# #     exit(0)
# camera_device = args.camera
# #-- 2. Read the video stream
# cap = cv.VideoCapture(camera_device)
# if not cap.isOpened:
#     print('--(!)Error opening video capture')
#     exit(0)
# # while True:
# #     ret, frame = cap.read()
# #     if frame is None:
# #         print('--(!) No captured frame -- Break!')
# #         break
# #     # cv.imshow("webcam", frame)
# #     # print("frame shape: %s , data type: %s" % (str(frame.shape), str(type(frame))))
# #     detectAndDisplay(frame)
# #     if cv.waitKey(17) == 27:
# #         break

# img = cv.imread("../res/img.png")


# while True:
#     detectAndDisplay(frame)

#     if cv.waitKey(17) == 27:
#         break
