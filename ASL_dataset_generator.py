import cv2
t = 0

cap = cv2.VideoCapture(0)

while (cap.read()):
    
    ret, frame = cap.read()
    frame1 = frame.copy()
    crop = frame[50:250,50:250]
    cv2.rectangle(frame1, (50,50), (250,250), (0,255,0), 1)
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
#     ret,crop = cv2.threshold(crop, 150,255,cv2.THRESH_BINARY_INV )
    cv2.imshow('Crop', crop)
    cv2.imshow('Frame', frame1)
    
    
    
    
    
    if cv2.waitKey(1) == ord('a'):
        print('Writing')
        for i in range(1,1001):
            ret, frame = cap.read()
            frame1 = frame.copy()
            crop = frame[50:250,50:250]
            cv2.rectangle(frame1, (50,50), (250,250), (0,255,0), 1)
            cv2.imshow('Frame', frame1)
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
#             ret,crop = cv2.threshold(crop, 150,255,cv2.THRESH_BINARY_INV )
            cv2.imshow('Crop', crop)
            crop = cv2.resize(crop,(30,30))
            cv2.imwrite('D:/Projects/Nivritti/ASL Temp/0/0' + str(i)+ '.png', crop)
            print(i)
            cv2.waitKey(1)
        print('Done !!!')
        break   
    
    
    
    
        t = 1
    if cv2.waitKey(1) == 27:
        break
    
    