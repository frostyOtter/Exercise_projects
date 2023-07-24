import numpy as np
import cv2
import tensorflow as tf

def img_preprocessing(img):
    img_ = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_ = cv2.resize(img_, (226, 226))
    _, img_ = cv2.threshold(img_, 122, 255, cv2.THRESH_BINARY)
    return np.array(img_)


if __name__ == "__main__":
    # init camera
    vidCap = cv2.VideoCapture(0)
    vidCap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    vidCap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    # init key map
    hand_getsure_dict = {
        0:'rock',
        1:'paper',
        2:'scissor',
        3:'nothing'
    }
    # load model:
    model = tf.keras.models.load_model('/Users/mike/SUMMER2023/DPL302m/RPS_project/rps_model_weight.h5')
    # turn on camera
    while vidCap.isOpened():
        isSuccess, frame = vidCap.read()
        frame = cv2.flip(frame, 1)
        # if camera is on
        if isSuccess:
            # init a rectangle to extract image
            cv2.rectangle(frame, (100, 100), (400, 400), (255, 102, 102), 2)
            # extract image
            img_extract = frame[100:400, 100:400]
            img_extract = img_preprocessing(img_extract)
            
            # predict from model
            pred = model.predict(np.expand_dims(img_extract, axis= 0))
            
            hand_getsure = hand_getsure_dict.get(np.argmax(pred))
            #cv2.putText(frame, hand_getsure)
            #cv2.putText(frame, hand_getsure, (0,0), cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=(0,0,255), lineType = cv2.LINE_AA)
            cv2.putText(frame, hand_getsure, (326, 326-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
        # cam show
        cv2.imshow('hand recognition', frame)
        if cv2.waitKey(1)&0xFF == 27:
            break
    
    vidCap.release()
    cv2.destroyAllWindows()