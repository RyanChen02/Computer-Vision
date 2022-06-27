import cv2
import mediapipe as mp
import time

#definiting variables of the hand and pose's features 
myDraw = mp.solutions.drawing_utils
myPose = mp.solutions.pose
mpDraw_hands = mp.solutions.drawing_utils
pose = myPose.Pose()
mpHands = mp.solutions.hands
hands = mpHands.Hands()


#inputting video's address
cap = cv2.VideoCapture("venv/Video-Import/Pexels Videos 2795730.mp4")


#while statement for module's iteration
while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results1= pose.process(imgRGB)

    if results1.pose_landmarks:
        myDraw.draw_landmarks(img, results1.pose_landmarks, myPose.POSE_CONNECTIONS)
        for id, lm in enumerate(results1.pose_landmarks.landmark):
            h, w, c = img.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(img, (cx, cy), 5, (650, 0, 0), cv2.FILE_NODE_STR)

#CPU processing time
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    
#setting image's frame  
    cv2.putText(img, str(int(fps)),(80,100),cv2.FONT_HERSHEY_PLAIN, 3,(500,2,1),3)
    cv2.imshow("Video_Recognition_Module_3 - Video 1", img)
    cv2.waitKey(1)
