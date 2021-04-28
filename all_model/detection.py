import cv2
import numpy as np

import time
net = cv2.dnn.readNet('obj_60000.weights', 'yolov3-tiny.cfg')

classes = ["car","truck","bus","motorcycle","auto","carLP","truckLP","busLP","motorcycleLP","autoLP"]

cap = cv2.VideoCapture("İsthanbul Evleri otopark bariyer sistemi.mp4")

font = cv2.FONT_HERSHEY_PLAIN
colors = np.random.uniform(0, 255, size=(100, 3))
(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
prev_frame_time = 0
  
# used to record the time at which we processed current frame
new_frame_time = 0
  
while True:
    _, img = cap.read()
    
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
    cv2.putText(img, fps, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)
    height, width, _ = img.shape

    blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0,0,0), swapRB=True, crop=False)
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
            
            if confidence > 0.2:
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
            cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
            """print(label)
            if(label=="Text5"):
                cropped = img[y-5:y+h+15,x-5:x+w+50]
               
                #reader = easyocr.Reader(['ar'], detection='DB', recognition = 'CNN_Transformer')
                reader = easyocr.Reader(['ar','en']) # need to run only once to load model into memory
                result = reader.readtext(cropped)
                print(result)
                cv2.imwrite("crooped.jpg",cropped)
                image_path=cv2.imread("cropped.jpg")
                out_image='outd.jpg'

                results=arabicocr.arabic_ocr(image_path,out_image)
                
                words=[]
                for i in range(len(results)):	
                        word=results[i][1]
                        words.append(word)
                print(str(words))"""
            cv2.putText(img, label + " " + confidence, (x, y-10), font, 2, (255,255,0), 1)
            """cropped = img[y-5:y+h+15,x-5:x+w]
           
            if(cropped.shape[0]>5 and cropped.shape[1]>5):
                cropped=cv2.resize(cropped,(300,300))
                gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
                ret,thresh = cv2.threshold(gray,75,255,cv2.THRESH_BINARY)
                blur = cv2.GaussianBlur(thresh,(5,5),0)                
                cv2.imshow('Binarization', thresh) 
                text1=pytesseract.image_to_string(thresh,lang="eng") 
                print(text1)
                                
            else:
                cv2.imshow('Image', img)"""
            

    cv2.imshow('Image', img)

    if cv2.waitKey (1) & 0xFF == ord ('q'):
        break
cap.release()
cv2.destroyAllWindows()
