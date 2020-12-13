import numpy as np
import cv2
import tensorflow as tf
m_new =tf.keras.models.load_model('D:\Yash\MyApplication\.vscode\digitrec.h5')
cl=False
a = np.ones([300,300],dtype='uint8')*255
win = 'Shapes'
img ='a'
cv2.namedWindow(win)

def digit(event,x,y,flags,param):
    global cl
    if event == cv2.EVENT_LBUTTONDOWN:
        cl=True
    if event == cv2.EVENT_MOUSEMOVE:
        if cl==True:
            cv2.rectangle(a,(x,y),(x+5,y+5),(0,0,0),-4)
    if event == cv2.EVENT_LBUTTONUP:
        cl=0
cv2.setMouseCallback(win,digit)

while True:
    cv2.imshow(win,a)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('c'):   
        a[:,:] = 255
    elif key == ord('p'):
        op = cv2.resize(a,(28,28)).reshape(-1,784)
        pre=np.argmax(m_new.predict(op))
        print(pre)
cv2.destroyAllWindows()