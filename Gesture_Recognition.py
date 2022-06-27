import cv2
import mediapipe as mp
import time

#definiting variables of the hand's features
mpDraw = mp.solutions.drawing_utils
mpDraw_hands = mp.solutions.drawing_utils
mpHands = mp.solutions.hands
hands = mpHands.Hands()

#definiting variables of the pose's features 
mpPose = mp.solutions.pose
pose = mpPose.Pose()


#opening the system camera
cap = cv2.VideoCapture(0)
pTime = 0


#definiting hand and gesture variables through while and if statement 
while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results1= pose.process(imgRGB)
    results2 = hands.process(imgRGB)

    if results1.pose_landmarks:
        mpDraw.draw_landmarks(img, results1.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(results1.pose_landmarks.landmark):
            h, w, c = img.shape
            print(id, lm)
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(img, (cx, cy), 5, (250, 0, 250), cv2.FILE_NODE_FLOAT)

            
    if results2.multi_hand_landmarks:
        for handLms in results2.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(img, (cx, cy), 5, (300, 0, 300), cv2.FILE_NODE_FLOAT)
            mpDraw_hands.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

            
#returning cpu processing time of module's running
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime


#setting image's frame
    cv2.putText(img, str(int(fps)), (100, 150), cv2.FONT_HERSHEY_PLAIN, 3,
                (300, 0, 0), 3)
    cv2.imshow("Pose_Recognition_Module 2", img)
    cv2.waitKey(1)
