# import imutils
# from imutils.video import VideoStreams
import argparse
import time
import cv2
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument(
    "-c",
    "--cascade",
    required=True,
    help=r"C:\Users\DIYA\Documents\SDP Project\Classsifier\haarcascade_frontalface_default.xml",
)
ap.add_argument(
    "-o", "--output", required=True, help=r"C:\Users\DIYA\Documents\SDP Project\data"
)
args = vars(ap.parse_args())

# # load OpenCV's Haar cascade for face detection from disk
# detector = cv2.CascadeClassifier(args["cascade"])
# # initialize the video stream, allow the camera sensor to warm up,
# # and initialize the total number of example faces written to disk
# # thus far
# print("[INFO] starting video stream...")
# vs = VideoStream(src=0).start()
# # vs = VideoStream(usePiCamera=True).start()
# time.sleep(2.0)
# total = 0
