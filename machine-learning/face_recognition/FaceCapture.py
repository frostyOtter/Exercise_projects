import cv2
from facenet_pytorch import MTCNN
import datetime
import os

# Initialize img path and input name
img_path = '/Users/mike/SPRING2023/CPV301/Assignment_CPV/db_face'
std_name = input('Input student name: ')
std_img_path = os.path.join(img_path, std_name)

# Number of img per student
count = 10

# fps for each img capture
leap = 1

# Initialize MTCNN modelvinh2

mtcnn = MTCNN(margin= 20, post_process=False, keep_all=False)

vidCap = cv2.VideoCapture(0)
vidCap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
vidCap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
while vidCap.isOpened() and count:
    isSuccess, frame = vidCap.read()
    if mtcnn(frame) is not None and leap%2:
        path = str(std_img_path+'/{}.jpg'.format(str(datetime.datetime.now())[:-7]+str(count)))
        face_img = mtcnn(frame, save_path = path)
        count-=1
    leap+=1
    cv2.imshow('Face Detection', frame)
    if cv2.waitKey(1) == 27:
        break
    
vidCap.release()
cv2.destroyAllWindows()