import cv2
import mediapipe as mp
import time

#importing cv2 module to open system camera
capture = cv2.Videoture(0)
hands = mp.solutions.hands
hands = hands.Hands()
layout = mp.solutions.drawing_utils


#setting process and cpu time to 0
process_time = 0
cpu_time = 0


#definiting hand's variables through while and if statement
while True:
    success, img = capture.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                len, high = int(lm.x * w), int(lm.y * h)
                print(id, len, high)
                cv2.circle(img, (len, high), 15, (300, 0, 300), cv2.FILE_NODE_FLOAT)
            hands.draw_landmarks(img, handLms, hands.HAND_CONNECTIONS)

            
#returning cpu processing time of module's running
    cpu_time = time.time()
    fps = 1 / (cpu_time - process_time)
    process_time = cpu_time
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 255), 3)

    
#setting image's frame 
    cv2.imshow("Hand_Recognition_Module", img)
    cv2.waitKey(1)
