print('Importing headers')
import cv2
import numpy as np
import pandas as pd

from keras import * 
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D,Conv1D,GlobalAveragePooling2D
from keras.utils import np_utils 
from keras.models import model_from_json
from keras.utils import to_categorical

from sklearn import preprocessing


le = preprocessing.LabelEncoder()

#################### For string labels encoding uncomment ################################
"""
le.fit(labels)
label1 = le.transform(labels)
label1 = to_categorical(label1,6)
"""
#################################                        ######################################

#################################### For reading test data ##############################
"""
print('Reading test images')
test = []
labels_test = np.array([1,1,1,1,1])

for i in range(1,101):
    img = cv2.imread('D:/Projects/Nivritti/ASL Test/Test' + str(i) + '.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.array(img).reshape(30,30,1)
    test.append(img)
    
test = np.array(test)
"""
#########################################################################################


print('Reading dataframe')

df = pd.read_csv('ASL_data_Numeric_labels_with_empty_backg_30x30_1000samp_number_only.csv') # Change the name as per your need

############# Extracting train data from the CSV file ##############################

labels = df.iloc[:, 0]

x_train = np.array(df.iloc[:, 1:(df.shape[1])])

x_train = x_train.reshape(11000,30,30,1)

print('Done !')

##################### Forming the architecture of deep learning model i.e CNN ################################

model = Sequential()

model.add(Conv2D(30, (3,3), input_shape=(30, 30, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
 
model.add(Conv2D(30, (3, 3), activation='relu'))
model.add(Conv2D(30, (3, 3), activation='relu')) 
model.add(Conv2D(30, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
 
model.add(Dropout(0.2))
model.add(Flatten())


model.add(Dense(1000, activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(12, activation='softmax'))

# Compile model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


  
print ('Training the model')
  
model.fit(x_train, labels,epochs = 10)

print('Done !!!!')

print ('Saving The classifier')
   
model_json = model.to_json()
with open("ASL_CNN_testing_model_for_four_samples_test_empty_backg_30x30_1000samp_numbers_only.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

print('Program Terminated')

######################### Related to test images block #################################
"""
print ('Evaluating models performance')
print('Predicting ....')

res = model.predict_classes(test)
print(res)

print('Program terminated')

"""
#########################################################################################

###################################################################################################################



############# For instant testing of the model ######################################
"""


print('Starting camera feed')
 
cap = cv2.VideoCapture(0)
 
while (cap.read()):
    cv2.waitKey(1) 
    ret, frame = cap.read()
    frame1 = frame.copy()
    crop = frame[50:250,50:250]
    crop = cv2.resize(crop, (30,30))
    
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    crop1 = crop.copy()
    
    crop = crop.reshape(1,30,30,1)
    
    res = model.predict_classes(crop)
    print(res)
    
    cv2.rectangle(frame1, (50,50), (250,250), (0,255,0), 1)
    cv2.imshow('Frame', frame1)
    cv2.imshow('Crop', crop1)
    
     
     
     
     
#     if cv2.waitKey(1) == ord('a'):
#         cv2.imshow('PRediction of image', crop1)
#         print('Predicting')
#         res = cnn.predict_classes(img)
#         print(res)
       
    if cv2.waitKey(1) == 27:
        break    
        
print('Program Terminated')       
"""
################################################################################


