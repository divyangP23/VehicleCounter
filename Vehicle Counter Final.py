import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *

cap = cv2.VideoCapture("../Videos/video.mp4")

model = YOLO('../Yolo-Weights/yolov8l.pt')

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train","truck", "jeep","helmet", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush","watch","plastic","scale","lock","key","autorikshaw","building","bridge","wheel"]

mask = cv2.imread("mask3.png")

limit = [300,350,673,350]
limit_2 = [300,400,673,400]

totalCount = []
objectFrameCount = {}
frameCount = 0
speed=0
maxSpeed = 0
minSpeed = float('inf')
totalSpeed = 0
vehicleCount = 0  # Variable to count the number of vehicles





frame_rate = cap.get(cv2.CAP_PROP_FPS)

tracker = Sort(max_age = 20, min_hits = 3, iou_threshold = 0.3)

while True:
    ret, frame = cap.read()
    frameRegion = cv2.bitwise_and(frame,mask)
    results = model(frameRegion, stream=True)
    detections = np.empty((0,5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1,y1,x2,y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1),int(y1),int(x2),int(y2)

            # the below is for fancy box
            w,h = x2-x1,y2-y1



            # this is for confidence meter
            conf = math.ceil((box.conf[0]*100))/100
            print(conf)

            # displaying confidence meter
            # cvzone.putTextRect(frame, f'{conf}', (max(0,x1),max(30,y1)))      #we have writen it below along with className so i have commented it out

            # this is for class names
            cls = int(box.cls[0])
            print(cls)

            # filtering specific vehicles
            if classNames[cls]=="car" or classNames[cls]=="motorbike" or classNames[cls]=="bus" or classNames[cls]=="truck" and conf > 0.3:
                # # this is design of boxes
                # cvzone.cornerRect(frame, (x1, y1, w, h), colorC=(0, 225, 0), l=10, rt=5)
                # # displaying class names
                # cvzone.putTextRect(frame, f'{classNames[cls]} {conf}', (max(0, x1), max(30, y1)), scale=0.9,
                #                    thickness=1, colorT=(255, 25, 203), colorR=(254, 180, 202), offset=5)

                # for tracking id
                currentArray = np.array([x1,y1,x2,y2,conf])
                detections = np.vstack((detections,currentArray))



            # the below is for normal box
            # print(x1,y1,x2,y2)
            # cv2.rectangle(frame,(x1,y1),(x2,y2),(200,0,200),4)

    resultsTracker = tracker.update(detections)
    cv2.line(frame,(limit[0],limit[1]),(limit[2],limit[3]),(0,255,255),5)  #this will draw a line
    cv2.line(frame,(limit_2[0],limit_2[1]),(limit_2[2],limit_2[3]),(255,0,255),5)  #this will draw a line
    for result in resultsTracker:
        x1,y1,x2,y2,Id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        print(result)
        w, h = x2 - x1, y2 - y1
        # this is design of boxes
        cvzone.cornerRect(frame, (x1, y1, w, h), colorC=(255, 255, 0), l=9, rt=3)
        # displaying id names
        cvzone.putTextRect(frame, f'{classNames[cls]} {conf}',
                           (max(0, x1), max(30, y1)), scale=1,
                           thickness=2, colorT=(255, 0, 0), colorR=(200, 180, 100), offset=10)

        cx , cy = x1+w//2 , y1+h//2  #this will define the center of box
        cv2.circle(frame,(cx,cy),5,(255,0,255),cv2.FILLED)

        # counting cars
        if limit[0]<cx<limit[2] and limit[1]-15<cy<limit[3]+15:
            if totalCount.count(Id)==0:
                totalCount.append(Id)
                cv2.line(frame, (limit[0], limit[1]), (limit[2], limit[3]), (255, 0, 255), 5) #this is for changing color after each detection
                objectFrameCount[Id] = frameCount
        if limit_2[0] < cx < limit_2[2] and limit_2[1] - 15 < cy < limit_2[3] + 15:
            if Id in objectFrameCount:
                entryTime = objectFrameCount[Id]
                timeTaken = frameCount - entryTime
                cv2.line(frame, (limit_2[0], limit_2[1]), (limit_2[2], limit_2[3]), (0, 255, 255), 5) #this is for changing color after each detection
                speed = 50/(timeTaken/frame_rate)

                if speed > maxSpeed:
                    maxSpeed = speed
                if speed < minSpeed:
                    minSpeed = speed

                totalSpeed += speed
                vehicleCount += 1


                # print(f"Vehicle {Id} took {timeTaken/frame_rate} seconds to cross from line 1 to line 2.")



    # Displaying count
    cvzone.putTextRect(frame, f'Count: {len(totalCount)}', (50,50), scale=2,
                           thickness=2, colorT=(255, 255, 255), colorR=(203, 192, 255), offset=10)
    # Displaying speed
    cvzone.putTextRect(frame, f'Speed: {speed}', (50,100), scale=2,
                           thickness=2, colorT=(255, 255, 255), colorR=(203, 192, 255), offset=10)
    # Displaying speed
    cvzone.putTextRect(frame, f'MaxSpeed: {maxSpeed}', (50, 150), scale=2,
                       thickness=2, colorT=(255, 255, 255), colorR=(203, 192, 255), offset=10)

    cvzone.putTextRect(frame, f'MinSpeed: {minSpeed}', (50, 200), scale=2,
                       thickness=2, colorT=(255, 255, 255), colorR=(203, 192, 255), offset=10)

    if vehicleCount > 0:
        averageSpeed = totalSpeed / vehicleCount
        cvzone.putTextRect(frame, f'AverageSpeed: {averageSpeed:.2f}', (50, 250), scale=2,
                           thickness=2, colorT=(255, 255, 255), colorR=(203, 192, 255), offset=10)

    cv2.imshow('frame', frame)
    # cv2.imshow('frameRegion', frameRegion)

    frameCount += 1


    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


#at last i am commenting out ever cvzone display so that it looks good if you want you can put it also