import cv2 as cv
from datetime import datetime
import os
from face_train import *
import pandas as pd


class harr_detector:
    def __init__(self):
        self.face_casecade = cv.CascadeClassifier()
        self.face_casecade.load("/Users/mike/SPRING2023/CPV301/Lab7/haarcascade_frontalface_alt.xml")
        # /Users/mike/SPRING2023/CPV301/Lab7/haarcascade_frontalface_default.xml
    def detect_face(self,img)->list:
        """
        trả về box cho các khuôn mặt trong hình
        :param img:
        :return:
        """
        img_gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        img_gray = cv.equalizeHist(img_gray)
        return self.face_casecade.detectMultiScale(img_gray)
    #idx, std_name = harr.face_recognition(face, meanface, std_name, ground_trues,local_log2, model)
    def face_recognition(self,img, local_meanface, local_label, local_groundtrues,local_log, model):
        results = {}
        for i, m_face in enumerate(local_meanface):

            result = model.show_result(m_face.reshape((160,160,3)).astype(np.uint8), img)
            if result is None:
                pass
            else:
                _, log2, _ = model.find_total50(result)
                test_score = np.abs(local_log[i]-log2)
                #test_score = (score/local_log[i])*100

                if 0 <= test_score <= local_groundtrues[i]:
                    results[local_label[i]] = test_score
                else: pass
        try:
            std_name_f = min(results, key = results.get)
        except:pass

        if len(results) == 0:
            return 'unknown'
        else:
            return std_name_f

    #Các hàm vẽ
    def draw_box(self,img,bboxs):
        for bbox in bboxs:
            (x, y, w, h) = bbox
            im = cv.rectangle(img, (x, y), (x+w, y+h), (255,102,102), 3)
        # for box in bboxs:
        #     (x, y, w, h) = box
        #     center=(x+w//2,y+h//2)
        #     im = cv.ellipse(img,center,(w//2,h//2),0,0,360,(255,102,102),2)
        return im


def extract_face(box, img):
    (x, y, w, h) = box
    face = img[y:y+h, x: x+w]
    face = cv2.resize(face, (160, 160), interpolation = cv2.INTER_AREA)
    return face

def load_data(path):
    df = pd.read_csv(path)
    meanface = df.iloc[:,0:-3].values
    std_name = df.iloc[:,-3].values
    ground_trues = df.iloc[:, -2].values
    local_log2 = df.iloc[:, -1].values
    return meanface, std_name, ground_trues, local_log2

if __name__ == "__main__":
    # initialize the library
    path = '/Users/mike/SPRING2023/CPV301/Assignment_CPV/local_db.csv'
    harr = harr_detector()
    model = sift_model()
    meanface_, std_name_, ground_trues_, local_log2_ = load_data(path)

    # turn on the camera
    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FRAME_WIDTH,640)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT,480)

    while cap.isOpened():
        isSuccess, frame = cap.read()
        if isSuccess:
            bboxs = harr.detect_face(frame)
            if bboxs is not None:
                for bbox in bboxs:
                    # frame = harr.draw_box(frame,bboxs)
                    box = list(map(int,bbox.tolist()))
                    face = extract_face(box, frame)
                    # face = face_array.reshape((160,160,3))
                    std_name = harr.face_recognition(face, meanface_, std_name_, ground_trues_,local_log2_, model)
                    frame = cv2.rectangle(frame, (box[0],box[1]), (box[0]+box[2],box[1]+box[3]), (255,102,102), 2)
                    frame = cv2.putText(frame, std_name, (box[0],box[1]), cv2.FONT_HERSHEY_DUPLEX, 2, (0,255,0), 2, cv2.LINE_8)

        cv.imshow('Face Recognition', frame)
        if cv.waitKey(1)&0xFF == 27:
            break

    cap.release()
    cv.destroyAllWindows()