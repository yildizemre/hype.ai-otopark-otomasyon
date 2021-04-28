import cv2
import numpy as np

import time
import sys

CONFIDENCE = 0.5
SCORE_THRESHOLD = 0.1
IOU_THRESHOLD = 0.1
config_path = "yolov3-tiny.cfg"
weights_path = "obj_60000.weights"
font_scale = 1
thickness = 1
labels = []

#classes_true = ["car","bicycle","motorbike","bus","truck","traffic light","stop sign"]
# with open("coco.names", "r") as f:
#     labels = [line.strip() for line in f.readlines()]
# labels = open("coco.names").read().strip().split("\n")
colors = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")
labels = ["car","truck","bus","motorcycle","auto","carLP","truckLP","busLP","motorcycleLP","autoLP"]
font = cv2.FONT_HERSHEY_PLAIN
net = cv2.dnn.readNetFromDarknet(config_path, weights_path)

ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
# read the file from the command line

cap = cv2.VideoCapture("videoplayback.mp4")
_, image = cap.read()
h, w = image.shape[:2]
fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter("videoplayback.avi", fourcc, 20.0, (w, h))

prev_frame_time = 0
  
# used to record the time at which we processed current frame
new_frame_time = 0
while True:
    _, image = cap.read()

    new_frame_time = time.time()
  
    # Calculating the fps
  
    # fps will be number of frame processed in given time frame
    # since their will be most of time error of 0.001 second
    # we will be subtracting it to get more accurate result
    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time
  
    # converting the fps into integer
    fps = int(fps)
  
    # converting the fps to string so that we can display it on frame
    # by using putText function
    fps = str(fps)
  
    # puting the FPS count on the frame
    cv2.putText(image, "fps"+fps, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)

    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (320, 320), swapRB=True, crop=False)
    net.setInput(blob)
    start = time.perf_counter()
    layer_outputs = net.forward(ln)
    time_took = time.perf_counter() - start
    print("Time took:", time_took)
    boxes, confidences, class_ids = [], [], []

    # loop over each of the layer outputs
    for output in layer_outputs:
        # loop over each of the object detections
        for detection in output:
            # extract the class id (label) and confidence (as a probability) of
            # the current object detection
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            # discard weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > CONFIDENCE:
                # scale the bounding box coordinates back relative to the
                # size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the boxes' width and height
                box = detection[:4] * np.array([w, h, w, h])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates, confidences,
                # and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # perform the non maximum suppression given the scores defined before
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, SCORE_THRESHOLD, IOU_THRESHOLD)

    font_scale = 1
    thickness = 1

    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            #if labels[class_ids[i]] in classes_true:
            # extract the bounding box coordinates
            x, y = boxes[i][0], boxes[i][1]
            w, h = boxes[i][2], boxes[i][3]
            # draw a bounding box rectangle and label on the image
            color = (0,255,255)
            cv2.rectangle(image, (x, y), (x + w, y + h), color=color, thickness=thickness)
            text = labels[class_ids[i]]
            # calculate text width & height to draw the transparent boxes as background of the text
            (text_width, text_height) = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, thickness=thickness)[0]
            text_offset_x = x
            text_offset_y = y - 5
            box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height))
            overlay = image.copy()
            cv2.rectangle(overlay, box_coords[0], box_coords[1], color=color, thickness=cv2.FILLED)
            # add opacity (transparency to the box)
            image = cv2.addWeighted(overlay, 0.6, image, 0.4, 0)
            # now put the text (label: confidence %)
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=font_scale, color=(0, 0, 0), thickness=thickness)
            try:
                if(text=="carLP"):
                    plaka=image[y:y+h,x:x+w]
                    cv2.imshow("car lp",plaka)

                    image[10:y+h,10:x+w]=plaka
                    
                    

            except:
                pass
            
    out.write(image)
    cv2.imshow("image", image)
    
    if ord("q") == cv2.waitKey(1):
        break


cap.release()
cv2.destroyAllWindows()