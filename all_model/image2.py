import cv2
import numpy as np
# from tensorflow import keras
import time
# import pytesseract
import os
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
# import arabic_ocr
# import arabicocr
# Match contours to license plate or character template
from tensorflow import keras
model = load_model('my_model.hdf5')
start = time.perf_counter()
# model = keras.models.load_model('model_segmentation.hdf5')
def find_contours(dimensions, img) :

    # Find all contours in the image
    cntrs, _ = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Retrieve potential dimensions
    lower_width = dimensions[0]
    upper_width = dimensions[1]
    lower_height = dimensions[2]
    upper_height = dimensions[3]
    
    # Check largest 5 or  15 contours for license plate or character respectively
    cntrs = sorted(cntrs, key=cv2.contourArea, reverse=True)[:15]
    
    ii = cv2.imread('contour.jpg')
    
    x_cntr_list = []
    target_contours = []
    img_res = []
    for cntr in cntrs :
        # detects contour in binary image and returns the coordinates of rectangle enclosing it
        intX, intY, intWidth, intHeight = cv2.boundingRect(cntr)
        
        # checking the dimensions of the contour to filter out the characters by contour's size
        if intWidth > lower_width and intWidth < upper_width and intHeight > lower_height and intHeight < upper_height :
            x_cntr_list.append(intX) #stores the x coordinate of the character's contour, to used later for indexing the contours

            char_copy = np.zeros((44,24))
            # extracting each character using the enclosing rectangle's coordinates.
            char = img[intY:intY+intHeight, intX:intX+intWidth]
            char = cv2.resize(char, (20, 40))
            
            cv2.rectangle(ii, (intX,intY), (intWidth+intX, intY+intHeight), (50,21,200), 2)
            

            # Make result formatted for classification: invert colors
            char = cv2.subtract(255, char)

            # Resize the image to 24x44 with black border
            char_copy[2:42, 2:22] = char
            char_copy[0:2, :] = 0
            char_copy[:, 0:2] = 0
            char_copy[42:44, :] = 0
            char_copy[:, 22:24] = 0

            img_res.append(char_copy) # List that stores the character's binary image (unsorted)
            
    # Return characters on ascending order with respect to the x-coordinate (most-left character first)
            
    plt.show()
    # arbitrary function that stores sorted list of character indeces
    indices = sorted(range(len(x_cntr_list)), key=lambda k: x_cntr_list[k])
    img_res_copy = []
    for idx in indices:
        img_res_copy.append(img_res[idx])# stores character images according to their index
    img_res = np.array(img_res_copy)

    return img_res
# Find characters in the resulting images
def segment_characters(image) :

    # Preprocess cropped license plate image
    img_lp = cv2.resize(image, (333, 75))
    img_gray_lp = cv2.cvtColor(img_lp, cv2.COLOR_BGR2GRAY)
    _, img_binary_lp = cv2.threshold(img_gray_lp, 200, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    img_binary_lp = cv2.erode(img_binary_lp, (3,3))
    img_binary_lp = cv2.dilate(img_binary_lp, (3,3))

    LP_WIDTH = img_binary_lp.shape[0]
    LP_HEIGHT = img_binary_lp.shape[1]

    # Make borders white
    img_binary_lp[0:3,:] = 255
    img_binary_lp[:,0:3] = 255
    img_binary_lp[72:75,:] = 255
    img_binary_lp[:,330:333] = 255

    # Estimations of character contours sizes of cropped license plates
    dimensions = [LP_WIDTH/6,
                       LP_WIDTH/2,
                       LP_HEIGHT/10,
                       2*LP_HEIGHT/3]
    
    

    # Get contours within cropped license plate
    char_list = find_contours(dimensions, img_binary_lp)

    return char_list


def fix_dimension(img): 
  new_img = np.zeros((28,28,3))
  for i in range(3):
    new_img[:,:,i] = img
  return new_img
  
def show_results():
    dic = {}
    characters = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    for i,c in enumerate(characters):
        dic[i] = c

    output = []
    for i,ch in enumerate(char): #iterating over the characters
        img_ = cv2.resize(ch, (28,28), interpolation=cv2.INTER_AREA)
        img = fix_dimension(img_)
        img = img.reshape(1,28,28,3) #preparing image for the model
        y_ = model.predict_classes(img)[0] #predicting the class
        character = dic[y_] #
        output.append(character) #storing the result in a list
        
    plate_number = ''.join(output)
    
    return plate_number

net = cv2.dnn.readNet('obj_60000.weights', 'yolov3-tiny.cfg')

classes = ["car","truck","bus","motorcycle","auto","carLP","truckLP","busLP","motorcycleLP","autoLP"]

cap = cv2.VideoCapture("İsthanbul Evleri otopark bariyer sistemi.mp4")

font = cv2.FONT_HERSHEY_PLAIN
colors = np.random.uniform(0, 255, size=(100, 3))
(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
prev_frame_time = 0
  
# used to record the time at which we processed current frame
new_frame_time = 0
  
img=cv2.imread("77EA287-11-40-49.jpg")
img = img.astype("uint8")
    #_, img = cap.read()
org_img=img.copy()
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
        
        if confidence > 0.3:
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
       
        ropped = org_img[y:y+h,x-10:x+w]

        if(label=="carLP"):
            
            cropped = org_img[y:y+h,x-10:x+w]
            
            char = segment_characters(cropped)
            text=show_results()
            cv2.putText(img, text, (x, y+h+50), font, 2, (255,255,0), 1)
            # config = '--oem 3 -l eng --psm 6 tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            # text2 = pytesseract.image_to_string(gray2, config=config)
            # text2 = "".join(text2.split()).replace(":", "").replace("-", "")
            # print("text2",text2)


            # text1=text.strip
          
            # text1=pytesseract.image_to_string(blur, 
            #                                 config = f'--psm 8 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
            # filter_predicted_result = "".join(text1.split()).replace(":", "").replace("-", "") 
            # print(filter_predicted_result)
            # cv2.imshow('Binarization', blur) 
            # #cv2.putText(img, text, (x, y+h+50), font, 2, (255,255,0), 1)
            # cv2.putText(img, filter_predicted_result, (x, y+h+50), font, 2, (255,255,0), 1)
            # # cv2.putText(img, text2, (x, y+h+100), font, 2, (255,255,0), 1)























            

            # reader = easyocr.Reader(['en']) # need to run only once to load model into memory
            # result = reader.readtext(cropped)
            # print("easyocr",result)





            # config = '-l eng --oem 1 --psm 7'
            # 

            # gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
            # ret,thresh = cv2.threshold(gray,150,255,cv2.THRESH_BINARY)
            # blur = cv2.GaussianBlur(thresh,(5,5),0)                
            
            
        # if(label=="Text5"):
        #     # cropped = img[y-5:y+h+15,x-5:x+w+50]
            
        #     #reader = easyocr.Reader(['ar'], detection='DB', recognition = 'CNN_Transformer')
        #     reader = easyocr.Reader(['ar','en']) # need to run only once to load model into memory
        #     result = reader.readtext(cropped)
        #     print(result)
        #     cv2.imwrite("crooped.jpg",cropped)
        #     image_path=cv2.imread("cropped.jpg")
        #     out_image='outd.jpg'

        #     results=arabicocr.arabic_ocr(image_path,out_image)
            
        #     words=[]
        #     for i in range(len(results)):	
        #             word=results[i][1]
        #             words.append(word)
        #     print(str(words))
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
        
        
time_took = time.perf_counter() - start
print("Time took:", time_took)
cv2.imshow('Image', img)
# filename = "{}.png".format(os.getpid())
# cv2.imwrite(filename, img)
cv2.waitKey(0)
    # if cv2.waitKey (1) & 0xFF == ord ('q'):
    #     break
cap.release()
cv2.destroyAllWindows()
