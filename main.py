import cv2
from cvzone.HandTrackingModule import HandDetector

# Webcam
cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

# Hand Detector
detector = HandDetector(maxHands = 1, detectionCon=0.8)


while True:
    # Get image frame
    success, img = cap.read()
    # Find the hand and its landmarks
    hands, img = detector.findHands(img)
    
    # landmark values = [x,y,z] * 21
    if hands:
        # get the first hand detected
        hand1 = hands[0]
        lmList1 = hand1["lmList"] # List of 21 Landmark points
    
    
    cv2.imshow("Image", img)
    cv2.waitKey(1)
    


