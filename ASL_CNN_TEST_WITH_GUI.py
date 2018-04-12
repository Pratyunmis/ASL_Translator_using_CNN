print('Importing headers')

##### For gui purpose

import tkinter as tk

####################

from keras.models import model_from_json
import cv2
import numpy as np
import pandas as pd
from scipy import misc
from sklearn.preprocessing import StandardScaler

print('Done')

##### Creating a Tkinter object for gui

root = tk.Tk()

root.title("Recognised Number")

back = tk.Frame(root, width=300, height=60)
back.pack()

tk.Label(root, 
         text="Recognised Digit:",
         fg = "black",
         font = "Times 40").pack()

lab = tk.Label(root, 
         text="Empty !!!",
         fg = "black",
         font = "Times 70")

lab.pack()
# root.update_idletasks()
# root.update()
# root.mainloop()

##############

print('Loading the model')

json_file = open('ASL_CNN_testing_model_for_four_samples_test_empty_backg_30x30_1000samp_numbers_only.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
cnn = model_from_json(loaded_model_json)
cnn.load_weights("model.h5")

print('Loaded model from disk')


############################################# For testing Test Images ################################
"""

img = cv2.imread('ASL Temp/D/D1.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = np.array(img).reshape(30,30,1)
test.append(img)
test = np.array(test)
print('Predicting img')
res = cnn.predict_classes(test)
print(res)
"""


#######################################################################################################

############################################# For testing a batch of Test Images ################################

"""
print('Reading test images')
test = []
labels = np.array([1,1,1,1,1])

for i in range(1,101):
    img = cv2.imread('D:/Projects/Nivritti/ASL Test/Test' + str(i) + '.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.array(img).reshape(30,30,1)
    test.append(img)
    
test = np.array(test)

print('Predicting ....')

res = cnn.predict_classes(test)
print(res)
"""

#######################################################################################################



print('Starting camera feed')
 
cap = cv2.VideoCapture(0)
 
while (cap.read()):
    cv2.waitKey(1) 
    ret, frame = cap.read()
    frame1 = frame.copy()
    crop = frame[50:250,50:250]
    crop1 = crop.copy()
    crop = cv2.resize(crop, (30,30))
    
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    
    
    crop = crop.reshape(1,30,30,1)
    
    res = cnn.predict_classes(crop)
    
    
    ########################### Change the representation of result according to your need #########################
    
   
    
    if(res == 10):
        print('Empty !!!')
        ##### For gui
        lab.config(text= 'Empty !!!')

        root.update_idletasks()
        root.update()
        
        
    else:
        print(res[0])
        
        ##### For gui
        lab.config(text=str(res[0]))

        root.update_idletasks()
        root.update()
    ###############################################################################################################       
    
    cv2.rectangle(frame1, (50,50), (250,250), (0,255,0), 1)
    cv2.imshow('Frame', frame1)
    cv2.imshow('Crop', crop1)
    
    
    
    ####################### Press Escape to Exit ######################################
    if cv2.waitKey(1) == 27:
        break    
     
     
############ Ignore this ######################################     
"""     
#     if cv2.waitKey(1) == ord('a'):
#         cv2.imshow('PRediction of image', crop1)
#         print('Predicting')
#         res = cnn.predict_classes(img)
#         print(res)
"""
################################################################
       
    
        
        
   