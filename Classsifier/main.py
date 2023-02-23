import cv2

def Detect(img, faceCasecade):
    color = {"blue":(255,0,0), "green":(0,255,0), "red":(0,0,255)}
    coords, img = draw_boundary(img, faceCascade, 1.1, 10, color["blue"], "Face")
    return img

# faceCascade = cv2.CasecadeClassifier("C:\Users\DIYA\Documents\SDP Project\Classsifier\haarcascade_frontalface_default.xml")

video_capture = cv2.VideoCapture(0)
while True:
    _, img = video_capture.read()
    # img = Detect(img, faceCascade)
    cv2.imshow("face detection",img)
    if cv2.waitKey(1) & 0xFF == ord('q') :
        break

video_capture.release()
cv2.destroyAllWinidows()