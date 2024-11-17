import cv2 as cv
import numpy as np

cap = cv.VideoCapture('object_video.mp4')

def rescaleframe(frame, scale=0.25): #Resize the frame 
    width = int(frame.shape[1]*scale)
    height = int(frame.shape[0]*scale)
    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

while True: 
    ret, frame = cap.read()
    if not ret:
        break
    frame_resized = rescaleframe(frame)

    #turn BGR to HSV
    hsv_frame = cv.cvtColor(frame_resized, cv.COLOR_BGR2HSV)
    lower_color = np.array([170,120,70]) #detect the object color
    high_color = np.array([180,255,255])

    #Masking
    mask = cv.inRange(hsv_frame, lower_color, high_color)
    masked = cv.bitwise_and(frame_resized, frame_resized, mask=mask)

    #object detection 
    contours, hier = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    print(f'Generate contours: {len(contours)} contours found!')

    for contour in contours:
        #calculate area 
        area = cv.contourArea(contour)
        if area > 100:
            cv.drawContours(frame_resized, [contour], -1, (0, 255, 0), 2)
            x, y, w, h = cv.boundingRect(contour)
            cv.rectangle(frame_resized, (x,y), (x + w, y + h), (0, 255, 0), 2)
            
    cv.imshow("Frame", frame_resized)
    cv.imshow("Mask_color", masked)
    cv.imshow('Mask',mask)

    if cv.waitKey(5) & 0xFF == ord('d'):
        break

cap.release()
cv.destroyAllWindows()
