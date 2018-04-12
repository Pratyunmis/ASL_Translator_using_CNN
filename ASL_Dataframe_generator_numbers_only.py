import cv2
import numpy as np
import pandas as pd


labels = list([])
X_train = list([])

c = -1  # counter variable for labels

print('Genarating Dataframe ...........')



for l in [ '0', '1', '2', '3' ,'4', '5' ,'6' ,'7', '8', '9', 'Empty']: 
    c = c+1
    print('Preparing dataset')
    
    for i in range(1,1001):
        
        img = cv2.imread('D:/Projects/Nivritti/ASL Temp/'+ str(l) + '/' + str(l) + str(i) + '.png')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         ret, img = cv2.threshold(img,150,255,cv2.THRESH_BINARY)
        img = cv2.resize(img, (30,30))
#         cv2.imwrite('D:/Projects/Nivritti/ASL Temp/' + str(l) + '/' + str(l) + str(i) + '.png', img)
        labels.append(c)
        X_train.append(img)
        
print('Done !!!!!!!!!')


X_train = np.array(X_train)
# print(X_train.shape)
X_train = X_train.reshape(11000,900)
# print(X_train)
labels = np.array(labels)
# print(labels)
print('Preparing Dataframe')

result = pd.DataFrame({'label':labels}) 

# print(result.shape)

for i in range(1,901):
    result[str(i)] = X_train[:,i-1]
         
      
print('Shuffling Dataframe')

result = result.sample(frac=1).reset_index(drop=True)       
    
    

print('Done !!!')

print('Saving CSV file')
       
result.to_csv('ASL_data_Numeric_labels_with_empty_backg_30x30_1000samp_number_only.csv', index = False)      

print('Program Terminated')  