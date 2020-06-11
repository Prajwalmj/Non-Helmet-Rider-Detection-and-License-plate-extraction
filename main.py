# -*- coding: utf-8 -*-
"""
Created on Sat May  4 13:34:30 2019

@author: prajwal mj
"""
import os
global found

video_name= input('Enter the video number')
if video_name== '1':
	ip1 = 'Testing_1.mp4' #video file name
	ip2 = '_1_' #video number
	ip3 = 'images\\frame_1_300.jpg' #selected frame for detection
	ip4 = 'frame_1_300new.jpg' #detection
	ip5 = '1.jpg' #input for license crop
	ip6 = 'img1_test.jpg'
	ip7 = 'Rotated_1_test.jpg' #Rotated license plate
	ip8 = 'C:\\Users\\prajwal mj\\Anaconda3\\envs\\tensorflow\\Library\\bin' # pytesseract path
	ip9 = 'bright_1_test.jpg' # after increasing brightness


elif video_name== '2':
	ip1 = 'Testing_2.mp4' #video file name
	ip2 = '_2_' #video number
	ip3 = 'images\\frame_2_350.jpg' #selected frame for detection
	ip4 = 'frame_2_350new.jpg' #detection
	ip5 = '2.jpg' #input for license crop
	ip6 = 'img2_test.jpg'
	ip7 = 'Rotated_2_test.jpg' #Rotated license plate
	ip8 = 'C:\\Users\\prajwal mj\\Anaconda3\\envs\\tensorflow\\Library\\bin' # pytesseract path
	ip9 = 'bright_2_test.jpg' # after increasing brightness


#frame collection
def frame_collector():
    import os
    import cv2
    execution_path='C:\\Users\\prajwal mj\\.spyder-py3\\Project\\frame_collect\\'
    vidcap = cv2.VideoCapture(execution_path+ ip1)
    i = 0
    save = 0 #number of frames
    ret = True
    while ret and i < 374: # 231 frames
        ret, frame = vidcap.read()
        if i % 50 ==0: # reading every 20 frames
            cv2.imwrite('C:\\Users\\prajwal mj\\.spyder-py3\\Project\\frame_collect\\images\\frame'+ ip2 + str(i) +  '.jpg' ,frame)   
            save +=1 # save frame as JPEG file
        i += 1
    vidcap.release()
    print("Total number of frames= ",i)
    cv2.destroyAllWindows()
#frame_collector()

def obj_det():

    from imageai.Detection import ObjectDetection

    execution_path='C:\\Users\\prajwal mj\\.spyder-py3\\Project\\frame_collect'


    detector = ObjectDetection()
    detector.setModelTypeAsYOLOv3()
    detector.setModelPath('C:\\Users\\prajwal mj\\.spyder-py3\\Project\\frame_collect\\yolo.h5')
    detector.loadModel()


    #read only custom objects, bike, person
    custom_objects = detector.CustomObjects(person=True,  motorcycle=True )

    detections, objects_path = detector.detectCustomObjectsFromImage(custom_objects=custom_objects, 
                                                   input_image =os.path.join(execution_path, 
                                                   ip3),
                                                   output_image_path=os.path.join(execution_path,
                                                   ip4),minimum_percentage_probability=50,
                                                   extract_detected_objects =  True)

# detections - list of dictionary containg details of name, box points, probability
# box points - is a tuple, containing top left, right bottom co-ordinates of bounding box
    for eachObject, eachObjectPath in zip(detections, objects_path):
        print(eachObject["name"] , " : " , eachObject["percentage_probability"], " : ", eachObject["box_points"] )
        print("Object's image saved in " + eachObjectPath)
        print("--------------------------------")
frame_collector()
obj_det()     
        
#helmet detection

from Helmet_det.hel_test import hel_test
from Helmet_det.image_crop import crop

found = 1
import os


root=os.getcwd()
os.chdir('C:\\Users\\prajwal mj\\.spyder-py3\\Project\\Helmet_det')
found = hel_test(ip1, ip2, ip3, ip4, ip5, ip6, ip7, ip8, ip9)
print('........',found)
os.chdir(root)

if found !=1:
    print ("Helmet is not found!")

    #License plate detection
    def open_test():
        import os
        '''os.system('cd C:\\Users\\prajwal mj\\.spyder-py3\\Project\\prompt_open\\job.bat')'''
        os.system("cd C:\\Users\\prajwal mj\\darkflow & call activate tensorflow & python flow --model cfg/yolo-new.cfg --load -1125 --imgdir sample_img & python flow --model cfg/yolo-new.cfg --load -1125 --imgdir sample_img --json --threshold 0.5 & deactivate")
    open_test()
    
    #license plate extraction
    
    def image_testcopy():
        from PIL import Image
        import cv2
        import json
        import os
        src = "C:\\Users\\prajwal mj\\darkflow\\sample_img\\out\\"
        bound = open(os.path.join(src, 'person-2.json'))
        data = json.load(bound)
        xbr1 = data[0]['bottomright']['x']
        ybr1 = data[0]['bottomright']['y']
        xtl1 = data[0]['topleft']['x']
        ytl1 = data[0]['topleft']['y']
        img_temp = cv2.imread("C:\\Users\\prajwal mj\\darkflow\\sample_img\\person-2.jpg")
        cv2.imwrite("C:\\Users\\prajwal mj\\.spyder-py3\\Project\\license_crop\\"+ ip5, img_temp)
        img = Image.open("C:\\Users\\prajwal mj\\.spyder-py3\\Project\\license_crop\\"+ ip5)
        #img2 = img.crop((220,455,317,512))
        #img2.save("C:\\Users\\prajwal mj\\.spyder-py3\\Project\\license_crop\\img7_t.jpg")
        #img3 = img.crop((xtl1+7,ytl1+16,xbr1-2,ybr1-12))                   #for testing 7
        img3 = img.crop((xtl1+29,ytl1+31,xbr1-15,ybr1-5))                  #for testing 3
        img3.save("C:\\Users\\prajwal mj\\.spyder-py3\\Project\\license_crop\\"+ ip6)
        print("Reached!!!")
    image_testcopy()
    
    #Pre-processing and character extraction
    
    def rotate():
        
        import pytesseract
        from PIL import Image
        import cv2
        import imutils
        
        image = cv2.imread("C:\\Users\\prajwal mj\\.spyder-py3\\Project\\license_crop\\"+ ip6,1)
        #image = cv2.imread("C:\\Users\\prajwal mj\\.spyder-py3\\license_crop\\img1_2.jpg")
        rotate = imutils.rotate_bound(image, 6)
        cv2.imwrite(ip7, rotate)
        #rescaling, because ocr was not recognizing. 110 x 66, was working
        oriimg = rotate
        height, width, depth = oriimg.shape
        imgScale = 1.47                                                #1.47 for testing 3
        newX,newY = oriimg.shape[1]*imgScale, oriimg.shape[0]*imgScale
        newimg = cv2.resize(oriimg,(int(newX),int(newY)))
        
        #incresing brightness
        hsv = cv2.cvtColor(newimg, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        value = 30
        lim = 255 - value
        v[v > lim] = 255
        v[v <= lim] += value
    
        final_hsv = cv2.merge((h, s, v))
        img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        #img = cv2.imread("C:\\Users\\prajwal mj\\Desktop\\mask_rcnn\\Mask_RCNN-master\\samples\\unmask\\motorcycle-1_0_licence.jpg")
        cv2.imwrite("C:\\Users\\prajwal mj\\Anaconda3\\envs\\tensorflow\\Library\\bin\\"+ ip9,img)
        result = pytesseract.image_to_string(Image.open("C:\\Users\\prajwal mj\\Anaconda3\\envs\\tensorflow\\Library\\bin\\"+ ip9))
        print("-----")
        print(result)
        print("----")
        result1 = []
        result = result.upper()
        for i in range(len(result)):
            char = result[i]
            if char.isalnum():
                result1.append(char)
    #print (result1)
        print(''.join(result1))
    rotate()
else: 
    print("Helmet found! No need for further processing")
