from ultralytics import YOLO
import cv2 as cv
import cvzone
import math

model = YOLO("yolov8n.pt")
cap = cv.VideoCapture('Video_5.mp4') #Video
#cap = cv.VideoCapture(1) #webcam

def rescaleframe(frame, scale=1.2): #Resize the frame 
    width = int(frame.shape[1]*scale)
    height = int(frame.shape[0]*scale)
    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_CUBIC)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_resized = rescaleframe(frame) #resized frame

    results = model(frame_resized, stream=True) 
    
    for r in results:
        boxes = r.boxes
        for box in boxes:

            #Bounding box
            x1,y1,x2,y2 = box.xyxy[0]
            x1,y1,x2,y2 = int(x1), int(y1), int(x2), int(y2) #convert to int

            w,h = x2-x1,y2-y1
            cvzone.cornerRect(frame_resized, (x1,y1,w,h))

            #confidence
            conf = math.ceil((box.conf[0]*100))/100

            #class name
            cls = int(box.cls[0])
            cvzone.putTextRect(frame_resized, f'{model.names[cls]}, 'f'{conf}',(max(0,x1), max(50,y1)), scale=0.7, thickness=1)

#Plot result
    cv.imshow('frame', frame_resized)
    if cv.waitKey(1) & 0xFF == ord('d'):
        break

cap.release()
cv.destroyAllWindows