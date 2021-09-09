import cv2
import numpy as np
import time
import HandTrackingModule as htm
import autopy
#########################
wCam, hCam = 640, 480
frameR = 100 #frame Redution
pTime = 0
plocX, plocY= 0,0
clocX, clocY,= 0,0
detector = htm.HandDetector(maxHands= 1)

wScr, hScr = autopy.screen.size()
print(wScr, hScr)
#########################
cap = cv2.VideoCapture(1)
cap.set(3,wCam)
cap.set(4,hCam)

while True:
    # 1. find hand landmarks
    success, img = cap.read()
    img = cv2.flip(img, 3)
    img = detector.findHands(img)
    lmList, bbox = detector.findPostion(img)
    print('bbox')
    print(bbox)



    # 2. get the tip of the index and middle fingers
    if len(lmList)!= 0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        #print(x1,y1)

        # 3. check with finger are up
        fingers = detector.fingersUp()
        print(fingers)
        # 4. only index finger : moving mode
        if fingers[1]==1 and fingers[2]==0:
            #5. convert coordicates
            cv2.rectangle(img, (frameR, frameR), (wCam-frameR, hCam-frameR), (255, 0, 255), 3)
            x3 = np.interp(x1, (frameR, wCam-frameR), (0,wScr))
            y3 = np.interp(y1, (0, hCam-frameR), (0, hScr))
            # 6. smoothen values

            clocX = plocX + (x3 -plocX) /smoothening
            clocY = plocY + (y3 -plocY) /smoothening

            # 7. move mouse
            autopy.mouse.move(clocX, clocY)
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            plocX, plocY = clocX, clocY
        # 8. Both finger and middle fingers are up : clicking mode
        if fingers[1] == 1 and fingers[2] == 1:
            # 9. find dictance between fingers
            length, img, lineinfo = detector.findDistance(8, 12, img)
            print(length)
            # 10. click mouse if dictance are short
            if length <40:
                cv2.circle(img, (lineinfo[4], lineinfo[5]), 15, (0,0,255), cv2.FILLED)
                autopy.mouse.click()





    # 11. frame rate

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (20,50), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 3)


    # 12. display


    cv2.imshow('results', img)
    cv2.waitKey(1)

