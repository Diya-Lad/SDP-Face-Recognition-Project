# draw_boundary()
import cv2


def generate_dataset(img, id, img_id):
    cv2.imwrite("data/user" + str(id) + "-" + str(img_id) + ".jpg", img)


# we are converting image into gray bcz complexity of gray color is lower compare to others
def draw_boundary(img, classifier, scalefactor, minNeighbors, color, text):
    # cvtColor is to convert one image to another color space
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # detectMultiscale - Detects objects of different sizes in the input image.
    # The detected objects are returned as a list of rectangles.
    # minNeighbors - this parameter will affect the quality of the detected faces.
    # Higher value results in less detections but with higher quality.
    # scaleFactor – Parameter specifying how much the image size is reduced at each image scale.
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = classifier.detectMultiScale(gray_img, scalefactor, minNeighbors)
    coords = []
    for x, y, w, h in features:
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        # id, _ = clf.predict(gray_img[y : y + j, x : x + w])
        # if id == 1:
        cv2.putText(
            img,
            text,
            (x, y - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            1,
            cv2.LINE_AA,
        )
        coords = [x, y, w, h]
    return coords


def recognize(img, clf, faceCascade):
    color = {"blue": (255, 0, 0), "green": (0, 255, 0), "red": (0, 0, 255)}
    coords = draw_boundary(img, faceCascade, 1.1, 10, color["red"], "Face", clf)
    return img


def Detect(img, faceCasecade, eyeCascade, noseCascade, mouthCascade, img_id):
    color = {"blue": (255, 0, 0), "green": (0, 255, 0), "red": (0, 0, 255)}
    coords = draw_boundary(
        img,
        faceCascade,
        1.1,
        2,
        color["blue"],
        "Face",
    )

    if len(coords) == 4:
        roi_img = img[
            coords[1] : coords[1] + coords[3], coords[0] : coords[0] + coords[2]
        ]
        # us_id_1 == 1
        user_id = 1
        generate_dataset(roi_img, user_id, img_id)

        # coords = draw_boundary(roi_img, eyeCascade, 1.1, 20, color["red"], "eye")
        # coords = draw_boundary(roi_img, noseCascade, 1.1, 15, color["green"], "nose")
        # coords = draw_boundary(roi_img, mouthCascade, 1.1, 20, color["red"], "mouth")

    return img


faceCascade = cv2.CascadeClassifier(
    r"C:\Users\DIYA\Documents\SDP Project\Classsifier\haarcascade_frontalface_default.xml"
)
eyeCascade = cv2.CascadeClassifier(
    r"C:\Users\DIYA\Documents\SDP Project\Classsifier\haarcascade_eye.xml"
)
noseCascade = cv2.CascadeClassifier(
    r"C:\Users\DIYA\Documents\SDP Project\Classsifier\Nariz.xml"
)
mouthCascade = cv2.CascadeClassifier(
    r"C:\Users\DIYA\Documents\SDP Project\Classsifier\Mouth.xml"
)

# clf=cv2.face.LBPHFaceRecognizer_create()
# clf.read("classifier.yml")

video_capture = cv2.VideoCapture(0)
imd_id = 0
while True:
    # video_capture is used to use webcam
    # read: is used to read a frame correctly
    _, img = video_capture.read()
    img_id = 1
    img = Detect(img, faceCascade, eyeCascade, noseCascade, mouthCascade, img_id)
    # img=recognize(img,clf,faceCascade)

    # cv2.imshow() method is used to display an image in a window
    cv2.imshow("face detection", img)
    img_id += 1
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWinidows()
