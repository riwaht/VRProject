import cv2
from cvzone.HandTrackingModule import HandDetector
import socket



# Parameters
widthCam, heightCam = 1280, 720
# Webcam
cap = cv2.VideoCapture(0)
cap.set(3, widthCam)
cap.set(4, heightCam)

# Hand Detector
detector = HandDetector(maxHands = 1, detectionCon=0.8)

# Communication
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
serverAddressPort = ("127.0.0.1", 5052)


while True:
    # Get image frame
    success, img = cap.read()
    # Find the hand and its landmarks
    hands, img = detector.findHands(img)
    
    data = []
    # landmark values = [x,y,z] * 21
    if hands:
        # get the first hand detected
        hand1 = hands[0]
        lmList1 = hand1['lmlist'] # List of 21 Landmark points
        for lm in lmList1:
             # append creates a separate list and appends the values in separate lines, we use extend to append the values in the same line
             # in open cv, the x and y values are reversed so we need to reverse them (they are in top left corner, unity is in bottom right corner)
             # we need to reverse the y value so that it is in the same direction as unity
            data.extend([lm[0], heightCam - lm[1], lm[2]])
        # send the data to unity
        sock.sendTo(str.encode(str(data)), serverAddressPort)
    
    img = cv2.resize(img, (0,0), None, 0.5, 0.5)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
    


