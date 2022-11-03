import cv2
import numpy as np
import mediapipe as mp
import time
from math import hypot
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# finger_distance_range from 18 to 412
# volume_range from -65.25 to 0.0

mphands = mp.solutions.hands
hands = mphands.Hands()
mpDraw = mp.solutions.drawing_utils
c4 = [0, 0]
c8 = [0, 0]
disList = []
vol = 0
volBar = 300
volPer = 0
def detectHand(img, imgRGB):
    result = hands.process(imgRGB)
    if result.multi_hand_landmarks:
        for handlm in result.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, handlm, mphands.HAND_CONNECTIONS)
            for id, lm in enumerate(handlm.landmark):
                h, w, c = img.shape
                if id == 4:
                    c4[0], c4[1] = int(lm.x*w), int(lm.y*h)
                    cv2.circle(img, (c4[0], c4[1]), 5, (0,255,0),-1)
                elif id ==8:
                    c8[0], c8[1] = int(lm.x*w), int(lm.y*h)
                    cv2.circle(img, (c8[0], c8[1]), 5, (0,255,0),-1)

    if len(c4) != 0 and len(c8) != 0:           
        cv2.line(img, (c4[0],c4[1]), (c8[0],c8[1]), (0,255,0), 3)  
        centerX, centerY = (c4[0]+c8[0])//2, (c4[1]+c8[1])//2 
        cv2.circle(img, (centerX, centerY), 5, (0,255,0), -1)
        distance = hypot(c4[0]-c8[0], c4[1]-c8[1])
        if distance > 0:
            disList.append(hypot(c4[0]-c8[0], c4[1]-c8[1]))
            print(max(disList), min(disList))
        if distance < 50:
            cv2.circle(img,(centerX, centerY), 8, (0,0,255), -1)
    return distance        
    
def volumeBar(img, dist):
    global volBar, volPer, vol
    cv2.rectangle(img, (50, 300), (80, 100), (255,0,0),5)
    if len(disList) != 0:
        volBar = np.interp(dist, [min(disList), max(disList)], [300, 100])
        volPer = np.interp(dist, [min(disList), max(disList)], [0, 100])
        vol = np.interp(dist, [min(disList), max(disList)], [-65.25, 0])
        print(f'volume{vol}')
    cv2.rectangle(img, (50, int(volBar)), (80,300), (255,0,0), -1)
    cv2.putText(img, f'{int(volPer)}%', (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,0,0), 3 )

def volumeSet():  
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(
        IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))
    # volume.GetMute()  # 靜音
    CurrentVol = volume.GetMasterVolumeLevel()
    VolRange = volume.GetVolumeRange()
    minVol = VolRange[0]
    maxVol = VolRange[1]
    # print(f'currentVol:{CurrentVol}')
    # print(f'maxVol:{maxVol}')
    # print(f'minVol:{minVol}')
    volume.SetMasterVolumeLevel(vol, None)


def main():
    cap = cv2.VideoCapture(0)
    ctime = 0
    ptime = 0
    while True:
        if cap.isOpened():
            success, img = cap.read()
            if not success:
                continue
        
            ctime = time.time()
            fps = 1/(ctime - ptime)
            ptime = ctime
            cv2.putText(img,str(int(fps)),(50,70), cv2.FONT_HERSHEY_COMPLEX, 2, (0,0,255), 5)

            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            dist = detectHand(img, imgRGB)
            volumeBar(img, dist)
            volumeSet()

            cv2.imshow('img', img)
            if cv2.waitKey(1) ==ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()