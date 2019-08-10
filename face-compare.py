from PIL import Image
from PIL import ImageFilter
import imagehash

img1=Image.open("190190_faces.jpg")
img2=Image.open("faces_detected.jpg")
if img1.width<img2.width:
    img2=img2.resize((img1.width,img1.height))
else:
    img1=img1.resize((img2.width,img2.height))
img1=img1.filter(ImageFilter.BoxBlur(radius=3))
img2=img2.filter(ImageFilter.BoxBlur(radius=3))
phashvalue=imagehash.phash(img1)-imagehash.phash(img2)
ahashvalue=imagehash.average_hash(img1)-imagehash.average_hash(img2)

totalaccuracy=phashvalue+ahashvalue
print(phashvalue)
print(ahashvalue)
print(totalaccuracy)
if(ahashvalue<30):
    print('no')
else:
    print('yes')
import numpy as np
import cv2

cap = cv2.VideoCapture(0)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        out.write(frame)
        cv2.imshow('frame',frame)
        cv2.imwrite('output.png',frame)
        img = cv2.imread('output.png',0)
        cv2.imshow('image',img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = faceCascade.detectMultiScale( img,scaleFactor=1.3,minNeighbors=3,minSize=(30, 30))
        print("[INFO] Found {0} Faces.".format(len(faces)))
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            roi_color = img[y:y + h, x:x + w]
            print("[INFO] Object found. Saving locally.")
            cv2.imwrite(str(w) + str(h) + '_faces.jpg', roi_color)
            status = cv2.imwrite('faces_detected.jpg', img)
            print("[INFO] Image faces_detected.jpg written to filesystem: ", status)

        break
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()