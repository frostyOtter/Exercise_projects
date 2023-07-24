import cv2
import os

if __name__ == "__main__":
    # Initialize img path and label name
    img_path = '/Users/mike/SUMMER2023/DPL302m/RPS_project/rps_img'
    label_name = input('Enter label name: ')
    label_img_path = os.path.join(img_path, label_name)
    print(label_img_path)
    # Number img per label
    count = 200

    # fps for each img capture
    leap = 1

    # Start and end key capture
    start = False

    # Init camera
    vidCap = cv2.VideoCapture(0)
    vidCap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    vidCap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while vidCap.isOpened() and count:
        isSuccess, frame = vidCap.read()
        frame = cv2.flip(frame, 1)
        # Init a rectangle to capture
        cv2.rectangle(frame, (100, 100), (400, 400), (255, 102, 102), 2)
        if start:
            # init path to save image captured
            save_path = os.path.join(label_img_path, '{}.jpg'.format(count))
            # capture images
            img_resize = cv2.resize(frame[100:400, 100:400], (226, 226))
            cv2.imwrite(str(save_path), img_resize)
            # show collecting text and img num
            #cv2.putText(frame, 'Collecting... {}'.format(count))
            #cv2.putText(frame, 'Collecting... {}'.format(count), (0,0), cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=(0,0,255), lineType = cv2.LINE_AA)
            cv2.putText(frame, 'Collecting...{}'.format(count), (326, 326-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
            count -= 1
        
        # show
        cv2.imshow('Hand capture', frame)
        # Init key to start and end collecting
        k = cv2.waitKey(10)
        if k == ord('s'):
            start = not start
        if k == ord('e'):
            break
        if count < 0:
            break

    vidCap.release()
    cv2.destroyAllWindows()


        