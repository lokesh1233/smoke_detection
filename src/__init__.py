import numpy as np
import tensorflow as tf
import cv2

model = tf.keras.models.load_model(
    'F:/studies/eclipse python/New folder3/jupyterNotebook/smoke_detection/models/save_mdl.h5',
    custom_objects = None,
    compile=True
)

video_src = 0
# video_src = "../datawe/raw/smoke_vdex1.mp4"
# video_src = "../datawe/raw/FreeSmokeYoutube.mp4"
# video_src = "../datawe/raw/nosmoke_vdex1.mp4"

#smoke detection opencv
fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
def opencv_frame(ret, frame):
    
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.GaussianBlur(gray_frame, (5,5), 5)
    fmask = fgbg.apply(gray_frame)
    kernel = np.ones((20, 20), np.uint8)
    fmask = cv2.medianBlur(fmask, 3)
    fmask = cv2.dilate(fmask, kernel)
    
    contours, hierarchy = cv2.findContours(fmask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(frame, (x, y), (x + w, y +h), (0, 255, 0), 2)
    cv2.imshow('frame', frame)



#video capture running

cap = cv2.VideoCapture(video_src)
i = 1
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    i += 1
    if frame is None:
        print("The End!")
        break
    
    if i > 200:
        break

    # Our operations on the frame come here
    #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
#     cv2.imshow('frame',frame)
    smoke_percent = round(model.predict([[cv2.resize(frame, (150, 150))/250]])[0][0], 3)
    if smoke_percent > 0.4:
        print('smoke detection: ', smoke_percent)
        opencv_frame(ret, frame)
    else:
        cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()