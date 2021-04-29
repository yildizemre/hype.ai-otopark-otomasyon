import cv2
import numpy as np
import easyocr
import time
import pytesseract
import os

start = time.perf_counter()
net = cv2.dnn.readNet('obj_60000.weights', 'yolov3-tiny.cfg')

classes = ["car","truck","bus","motorcycle","auto","carLP","truckLP","busLP","motorcycleLP","autoLP"]

font = cv2.FONT_HERSHEY_PLAIN
colors = np.random.uniform(0, 255, size=(100, 3))
(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
prev_frame_time = 0
  
new_frame_time = 0
  
img=cv2.imread("61EH553-13-16-29.jpg")

height, width, _ = img.shape

blob = cv2.dnn.blobFromImage(img, 1/255, (316, 316), (0,0,0), swapRB=True, crop=False)
net.setInput(blob)
output_layers_names = net.getUnconnectedOutLayersNames()
layerOutputs = net.forward(output_layers_names)

boxes = []
confidences = []
class_ids = []

for output in layerOutputs:
    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        
        if confidence > 0.1:
            center_x = int(detection[0]*width)
            center_y = int(detection[1]*height)
            w = int(detection[2]*width)
            h = int(detection[3]*height)

            x = int(center_x - w/2)
            y = int(center_y - h/2)

            boxes.append([x, y, w, h])
            confidences.append((float(confidence)))
            class_ids.append(class_id)
            #print(confidence)
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)

if len(indexes)>0:
    for i in indexes.flatten():
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence = str(round(confidences[i],2))
        #color = colors[i]
        #cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
       
        if(label=="carLP"):
            
            #cropped = img[y:y+h,x:x+w]
            resize_cropped = cv2.resize(
                    img[y:y+h,x:x+w], None, fx = 2, fy = 2, 
                    interpolation = cv2.INTER_CUBIC)
            gray = cv2.cvtColor(resize_cropped, cv2.COLOR_BGR2GRAY)
            ret,thresh = cv2.threshold(gray,75,255,cv2.THRESH_BINARY)
            blur = cv2.GaussianBlur(thresh,(5,5),0)
        
            text1=pytesseract.image_to_string(blur, 
                                            config = f'--psm 8 -l eng+tur --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
            filter_predicted_result = "".join(text1.split()).replace(":", "").replace("-", "")
            print(filter_predicted_result) 
            #cv2.putText(img, filter_predicted_result, (x, y+h+50), font, 2, (255,255,0), 1)
           
        #cv2.putText(img, label + " " + confidence, (x, y-10), font, 2, (255,255,0), 1)
        
time_took = time.perf_counter() - start
print("Time took:", time_took)
# cv2.imshow('Image', img)

# cv2.waitKey(0)
#     # if cv2.waitKey (1) & 0xFF == ord ('q'):
#     #     break
# cap.release()
# cv2.destroyAllWindows()
