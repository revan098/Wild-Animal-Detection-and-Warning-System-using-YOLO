comport='COM5'
import cv2 as cv
import argparse
import sys
import numpy as np
import os
import csv
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
import serial
import time

ser = serial.Serial(comport, 9600, timeout=1)
time.sleep(2) 
EMAIL_SENDER = "officialforproject@gmail.com"
EMAIL_PASSWORD = "wdatuzzhhtwjsroi"  
EMAIL_RECEIVER = "officialforproject@gmail.com"
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587

confThreshold = 0.5
nmsThreshold = 0.4
inpWidth = 416
inpHeight = 416
def send_email_alert(class_name, confidence):
    subject = f"Animal Detected: {class_name}"
    body = f"An animal was detected.\n\nType: {class_name}\nConfidence: {confidence:.2f}\nTime: {datetime.now().isoformat()}"

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = EMAIL_SENDER
    msg["To"] = EMAIL_RECEIVER

    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.send_message(msg)
            print(f"[INFO] Email alert sent for {class_name}")
    except Exception as e:
        print(f"[ERROR] Failed to send email: {e}")


classesFile = "coco.names"
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

modelConfiguration = "yolo.cfg"
modelWeights = "yolo.weights"
net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

def getOutputsNames(net):
    layersNames = net.getLayerNames()
    return [layersNames[i - 1] for i in net.getUnconnectedOutLayers()]

def drawPred(classId, conf, left, top, right, bottom):
    cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)
    label = '%s: %.2f' % (classes[classId], conf)
    labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv.rectangle(frame, (left, top - round(1.5 * labelSize[1])), (left + round(1.5 * labelSize[0]), top + baseLine), (255, 255, 255), cv.FILLED)
    cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1)
    ser.write((label + "\n").encode('utf-8'))


def logDetection(classId, conf, box):
    class_name = classes[classId]
    with open('detections.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([datetime.now().isoformat(), class_name, conf, *box])
    send_email_alert(class_name, conf)



def postprocess(frame, outs):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    classIds = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold and (14<classId<25) : 
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    current_frame_classes = set()

    for i in indices:
        box = boxes[i]
        drawPred(classIds[i], confidences[i], *box)

        class_name = classes[classIds[i]]
        current_frame_classes.add(class_name)
        if class_name not in logged_classes:
            logDetection(classIds[i], confidences[i], box)
            logged_classes.add(class_name)

    for cls in list(logged_classes):
        if cls not in current_frame_classes:
            logged_classes.remove(cls)

cap = cv.VideoCapture(0)
cv.namedWindow('Detection', cv.WINDOW_NORMAL)

logged_classes = set()

while cv.waitKey(1) < 0:
    hasFrame, frame = cap.read()
    if not hasFrame:
        break

    blob = cv.dnn.blobFromImage(frame, 1 / 255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)
    net.setInput(blob)
    outs = net.forward(getOutputsNames(net))
    postprocess(frame, outs)

    cv.imshow('Detection', frame)
cv.destroyAllwindows()
ser.close()