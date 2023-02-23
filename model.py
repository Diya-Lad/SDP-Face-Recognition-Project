# import numpy as np

# import cv2

# face_cascade = cv2.CascadeClassifier(
#     r"C:\Users\DIYA\Documents\SDP Project\Classsifier\haarcascade_frontalface_default.xml"
# )

# eye_cascade = cv2.CascadeClassifier(
#     r"C:\Users\DIYA\Documents\SDP Project\Classsifier\haarcascade_eye.xml"
# )

# cap = cv2.VideoCapture(0)
# while 1:
#     ret, img = cap.read()

#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#     faces = face_cascade.detectMultiScale(gray, 1.5, 5)

#     for x, y, w, h in faces:
#         cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

#         roi_gray = gray[y : y + h, x : x + w]

#         roi_color = img[y : y + h, x : x + w]

#         eyes = eye_cascade.detectMultiScale(roi_gray)

#         for ex, ey, ew, eh in eyes:
#             cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

#     print("found " + str(len(faces)) + " face(s)")

#     cv2.imshow("img", img)

#     k = cv2.waitKey(30) & 0xFF

#     if k == 27:
#         break

# cap.release()

# cv2.destroyAllWindows()

import numpy as np

import cv2

face_cascade = cv2.CascadeClassifier(
    r"C:\Users\DIYA\Documents\SDP Project\Classsifier\haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)

id = input("enter user id")

sampleN = 0

while 1:
    ret, img = cap.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for x, y, w, h in faces:
        sampleN = sampleN + 1

        cv2.imwrite(
            r"C:\Users\DIYA\Documents\SDP Project\data\temp"
            + str(id)
            + "."
            + str(sampleN)
            + ".jpg",
            gray[y : y + h, x : x + w],
        )

        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.waitKey(100)

    cv2.imshow("img", img)

    cv2.waitKey(1)

    if sampleN > 20:
        break

cap.release()

cv2.destroyAllWindows()
