"""
CV Mini Project ~ Eye Detection Using HaarCascade & Its Application in Visual Painting

* PF 28 Oishik Mandal
* PF 33 Lakshita Agarwal
* PF 40 Khuzema Khomosi
"""

# Built In libs for Utility
import os
import operator

# GUI Info Draw
import tkinter as tk
from tkinter import messagebox

# Will use to load HaarCascade, and feed video input
import cv2 as cv
# Numerical Window Based Calculations
import numpy as np
# Track Screen Size
from screeninfo import get_monitors

# Height & Width will allow us to set window dimensions
WINDOW_WIDTH = round(get_monitors()[0].width * 3 / 4)
WINDOW_HEIGHT = round(get_monitors()[0].height * 3 / 4)

# Basic Window Properties
WINDOW_NAME = "EyePaint"
TH_BAR_NAME = 'Eye Detection Threshold'

# Key Press Detection Timeout(ms)
KEY_DELAY = 33

# Hiding Root Window, since we render that via openCV
root = tk.Tk()
root.withdraw()

# Higher Threshold detects better in low light,
# however in optimal conditions, captures more ~ 42 is experimental best
PUPIL_THRESHOLD = 42
# Lower Overlapping threshold means faster results, since we are considering nearly similar as same
# however accuracy will drop
OVERLAP_THRESHOLD = 0.95
# Higher this number, more relaxed the cursor movement is
MOVE_THRESHOLD = 2
# PHASE 0: Pupils configuration
# PHASE 1: Eyes Calibration
# PHASE 2: Paint Mode
PHASE = 0

# Cursor Position to consider when calibrating
START_CURSOR_POS = [-1, -1]

# Cascade XML Inputs
HARR_FACE = r'cascade/haarcascade_frontalface_default.xml'
HARR_EYE = r'cascade/haarcascade_eye.xml'

if __name__ != '__main__':
    print("Please run this script as the main entrypoint.")
    exit()

cap = cv.VideoCapture(0)  # Capture Webcam Feed

if not cap.isOpened():
    print("Cannot open camera.")
    exit()

cv.namedWindow(WINDOW_NAME, cv.WINDOW_FULLSCREEN)  # Program Window Name

# We might need to change detection threshold based on lighting
cv.createTrackbar(TH_BAR_NAME, WINDOW_NAME, 0, 255, (lambda a: None))
cv.setTrackbarPos(TH_BAR_NAME, WINDOW_NAME, PUPIL_THRESHOLD)


class CascadeDetector:
    """
        This Class represents an instance of Haar Classifier.
        This classifier is able to analyse the input image looking for particular features
        that satisfy the provided model and will be used to identify face & eyes
    """

    def __init__(self):
        # Loading the pretrained cascade files
        self.face_cascade = cv.CascadeClassifier(HARR_FACE)
        self.eye_cascade = cv.CascadeClassifier(HARR_EYE)

        # Convert Image of Eyes into Blobs for recognition
        blob_detect_params = cv.SimpleBlobDetector_Params()
        blob_detect_params.filterByArea = True
        blob_detect_params.maxArea = 1500  # Experimental value, best area for pupils in face
        self.blobDetector = cv.SimpleBlobDetector_create(blob_detect_params)

        # Detection Threshold Parameter, defaults to best fit
        self.pupil_threshold = PUPIL_THRESHOLD
        # Overlap Threshold, defaults to best fit
        self.overlap_threshold = OVERLAP_THRESHOLD
        # Detection Calibration occurs in Phase 0
        self.phase = 0

        # Referential Variables
        self.face_frame = None
        self.previous_face = [0, 0, 0, 0]
        self.previous_left_eye = [-1, 0, 0, 0]
        self.previous_right_eye = [-1, 0, 0, 0]
        self.left_is_visible = False
        self.right_is_visible = False
        self.left_eye_frame = None
        self.right_eye_frame = None
        self.lp_frame = None
        self.rp_frame = None
        self.lp_thresh_frame = None
        self.rp_thresh_frame = None
        self.move_thresh = 0.4
        self.left_pupil = [0, 0]
        self.right_pupil = [0, 0]
        self.tmp_left_pupil = [0, 0]
        self.tmp_right_pupil = [0, 0]

    def startPhase(self, phase, threshold=OVERLAP_THRESHOLD):
        self.phase = phase
        self.overlap_threshold = threshold

    def findEyes(self, current_frame):
        # This function is culmination of others and detects pupils
        # First we update user set threshold
        self.updateThreshold()

        # Finding out details of frame, and backing up current image
        frame_width = current_frame.shape[1]
        frame_height = current_frame.shape[0]
        frame_ratio = frame_width / frame_height
        frame_original = cv.copyTo(current_frame, None)

        # Detecting all faces in current frame
        # ~ Will only work with a single face, multiple faces will cause excess blur
        faces = self.detectFace(current_frame)

        for (x, y, w, h) in faces:
            # Current Face Data
            face_width = int(frame_width / 3)
            face_height = int(face_width / frame_ratio)
            face_xcoord = int(x + w / 2 - face_width / 2)
            face_ycoord = int(y + h / 2 - face_height / 2)

            # We select face frame from entire picture and then stabilize it for blob detection
            self.face_frame = frame_original[face_ycoord:face_ycoord + face_height, face_xcoord:face_xcoord + face_width]
            x, y, w, h = self.stabilizeFaceFrame(x, y, w, h)

            # Drawing rectangle around face
            cv.rectangle(current_frame, (x, y), (x + w, y + h), (255, 255, 0), 2)

            # Blurring out rest of the background after face detection
            current_frame[0:y, 0:frame_width] = cv.GaussianBlur(current_frame[0:y, 0:frame_width], (0, 0), 4)
            current_frame[y:y + h, 0:x] = cv.GaussianBlur(current_frame[y:y + h, 0:x], (0, 0), 4)
            current_frame[y + h:frame_height, 0:frame_width] = cv.GaussianBlur(current_frame[y + h:frame_height, 0:frame_width], (0, 0), 4)
            current_frame[y:y + h, x + w:frame_width] = cv.GaussianBlur(current_frame[y:y + h, x + w:frame_width], (0, 0), 4)

            # Detecting Eyes in Current Face
            eyes = self.detectEyes(self.face_frame)

            # Default Invisible State
            self.left_is_visible = False
            self.right_is_visible = False

            # Iterate over detected eyes, classifying them as left or right
            for (ex, ey, ew, eh) in eyes:
                if (ey + eh) > (face_height / 2):
                    # Invalid Eye Location, eyes are above face
                    pass

                if (ex + ew / 2) < (face_width / 2):
                    # Since End of Eye is before face center, LEFT eye
                    self.left_is_visible = True
                    # Stabilize Eyes for Blob Detection
                    ex, ey, ew, eh, self.previous_left_eye = self.stabilizeEyeFrame(face_xcoord, face_ycoord, ex, ey, ew, eh, self.previous_left_eye)

                    # Rectangle over eyes
                    cv.rectangle(current_frame, (face_xcoord + ex, face_ycoord + ey), (face_xcoord + ex + ew, face_ycoord + ey + eh), (255, 0, 255), 2)

                    if self.phase > 0:
                        # Secondary face frame is only needed, when model is actively tracking face in phase 1 & 2
                        cv.rectangle(self.face_frame, (ex, ey), (ex + ew, ey + eh), (255, 0, 255), 2)

                    # Left Eye Frame & Pupil detection
                    self.left_eye_frame = self.face_frame[ey:ey + eh, ex:ex + ew]
                    lp_keypoint, lt_img = self.detectPupils(self.left_eye_frame, self.pupil_threshold)

                    # The points or spatial locations in a given image which defines whatever stands out in the image is called keypoint.
                    # Scale-invariant feature transform is used to detect features
                    # We use RICH keypoints so that size of circle corresponds to blob size
                    # Draw Keypoints over Possible Pupils
                    self.lp_thresh_frame = cv.cvtColor(lt_img, cv.COLOR_GRAY2BGR)
                    self.lp_frame = cv.copyTo(self.lp_thresh_frame, None)
                    self.lp_frame = cv.drawKeypoints(self.lp_frame, lp_keypoint, self.lp_frame, (0, 255, 0), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

                    if len(lp_keypoint) > 0:
                        # Eyes are detected by Blob
                        self.tmp_left_pupil = [int(lp_keypoint[0].pt[0]), int(lp_keypoint[0].pt[1])]
                        current_frame = cv.circle(current_frame, (face_xcoord + ex + self.left_pupil[0], face_ycoord + ey + self.left_pupil[1]), 5, (0, 255, 0), 4)
                        self.face_frame = cv.circle(self.face_frame, (ex + self.left_pupil[0], ey + self.left_pupil[1]), 5, (0, 255, 0), 4)

                    else:
                        # Eyes are not detected, hence we consider center of bounding box
                        self.tmp_left_pupil = [face_xcoord + ex + int(ew / 2), face_ycoord + ey + int(eh / 2)]

                else:
                    # Otherise End of Eye is after face center, RIGHT eye
                    self.right_is_visible = True
                    # Stabilize Eyes for Blob Detection
                    ex, ey, ew, eh, self.previous_right_eye = self.stabilizeEyeFrame(face_xcoord, face_ycoord, ex, ey, ew, eh, self.previous_right_eye)

                    # Rectangle over eyes
                    cv.rectangle(current_frame, (face_xcoord + ex, face_ycoord + ey), (face_xcoord + ex + ew, face_ycoord + ey + eh), (255, 0, 255), 2)

                    if self.phase > 0:
                        cv.rectangle(self.face_frame, (ex, ey), (ex + ew, ey + eh), (255, 0, 255), 2)

                    # Right Eye Frame & Pupil detection
                    self.right_eye_frame = self.face_frame[ey:ey + eh, ex:ex + ew]
                    rp_keypoint, rt_img = self.detectPupils(self.right_eye_frame, self.pupil_threshold)

                    # Draw Keypoints over Possible Pupils
                    self.rp_thresh_frame = cv.cvtColor(rt_img, cv.COLOR_GRAY2BGR)
                    self.rp_frame = cv.copyTo(self.rp_thresh_frame, None)
                    self.rp_frame = cv.drawKeypoints(self.rp_frame, rp_keypoint, self.rp_frame, (0, 255, 0), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

                    if len(rp_keypoint) > 0:
                        # Eyes are detected by Blob
                        self.tmp_right_pupil = [int(rp_keypoint[0].pt[0]), int(rp_keypoint[0].pt[1])]
                        current_frame = cv.circle(current_frame, (face_xcoord + ex + self.right_pupil[0], face_ycoord + ey + self.right_pupil[1]), 5, (0, 255, 0), 4)
                        self.face_frame = cv.circle(self.face_frame, (ex + self.right_pupil[0], ey + self.right_pupil[1]), 5, (0, 255, 0), 4)

                    else:
                        # Eyes are not detected, hence we consider center of bounding box
                        self.tmp_right_pupil = [face_xcoord + ex + int(ew / 2), face_ycoord + ey + int(eh / 2)]

        # Update Eye Locations before returning current face frame
        self.checkEyes()
        return current_frame

    def updateThreshold(self):
        # Updates Current Threshold value
        self.pupil_threshold = cv.getTrackbarPos(TH_BAR_NAME, WINDOW_NAME)

    def detectFace(self, bgr_image):
        # Converts the image to grayscale before detecting face via face_cascade
        gray_image = cv.cvtColor(bgr_image, cv.COLOR_BGR2GRAY)
        return self.face_cascade.detectMultiScale(gray_image, 1.3, 5)  # image, scale ~ magnify to 1.3, neighbours ~ 5

    def stabilizeFaceFrame(self, x, y, w, h):
        # In particular, the Euclidean distance of a vector from the 
        # origin is a norm, called the Euclidean norm. Hence we use this to find 
        # if the face has actually moved sufficiently to re-calculate pupil blobs
        # We first derive the previously normalized face location
        prev_norm = cv.norm(np.array([x, y, w, h], np.float32), np.array(self.previous_face, np.float32))

        # If the face has moved greater than threshold amount
        # we then update our previous face to current, else reset
        # Reason: Blob detection is not ideal for a real-time application as it is not always able to detect pupils
        if prev_norm > 60:
            self.previous_face = [x, y, w, h]

        else:
            x = self.previous_face[0]
            y = self.previous_face[1]
            w = self.previous_face[2]
            h = self.previous_face[3]

        return x, y, w, h

    def detectEyes(self, bgr_image):
        # Converts the image to grayscale before detecting eye via eye_cascade
        gray_image = cv.cvtColor(bgr_image, cv.COLOR_BGR2GRAY)
        return self.eye_cascade.detectMultiScale(gray_image, 1.3, 5)  # image, scale ~ magnify to 1.3, neighbours ~ 5

    def stabilizeEyeFrame(self, face_x, face_y, x, y, w, h, previous_eyes_coords):
        # Unlike face we need to compare a greater overlapping area for eye frame
        if previous_eyes_coords[0] == -1 or self.checkOverlapArea(face_x + x, face_y + y, w, h, previous_eyes_coords):
            previous_eyes_coords = [face_x + x, face_y + y, w, h]

        else:
            x = previous_eyes_coords[0] - face_x
            y = previous_eyes_coords[1] - face_y
            w = previous_eyes_coords[2]
            h = previous_eyes_coords[3]

        return x, y, w, h, previous_eyes_coords

    def checkOverlapArea(self, x, y, w, h, previous_eyes_coords):
        # Overlap Calulations are done based on distance of
        # eye from previously recorded center
        px = previous_eyes_coords[0]
        py = previous_eyes_coords[1]
        pw = previous_eyes_coords[2]
        ph = previous_eyes_coords[3]

        over_x1 = x if x < px else px
        over_y1 = y if y < py else py
        over_x2 = (x + w) if x + w > px + pw else px + pw
        over_y2 = (y + h) if y + h > py + ph else py + ph

        overlap_area = (over_x2 - over_x1) * (over_y2 - over_y1)
        actual_area = w * h

        # By using floating point calculation we check, how much overlap has occurred,
        # if the overlap is less than threshold, we update eyes to new coordinates
        overlap_rate = actual_area / overlap_area
        return overlap_rate < self.overlap_threshold

    def detectPupils(self, bgr_image, threshold=PUPIL_THRESHOLD):
        # Here we use blob detection to identify pupils
        img = cv.copyTo(bgr_image, None)
        # We crop the image and change color for the rest
        img[0:int(img.shape[0] / 4), 0:img.shape[1]] = (255, 255, 255)

        # First we convert the image to grayscale
        # Then we perform binary thresholding in the image
        # What threshold does, is for every gray value of pixel, should it be above
        # a certain threshold, it is upcolored to another value
        # For our case, for every pixel with grays above PUPIL_THRESHOLD, will be have its grayness set to 255
        gray_frame = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        _, t_img = cv.threshold(gray_frame, threshold, 255, cv.THRESH_BINARY)

        # Image Manipulation for better results
        # Erodes boundaries of image, diminishes features
        img = cv.erode(t_img, None, iterations=2)
        # Increases area, accentuate features, dialation after erosion, acts to smoothen image, removing grains
        img = cv.dilate(img, None, iterations=4)
        # Further Image Smoothening to prevent artifacts
        img = cv.medianBlur(img, 5)

        return self.blobDetector.detect(img), t_img

    def checkEyes(self):
        # Checks if the eyes have moved, if so update them
        # Actual checking is done via the same absolute normailization provided via openCV
        right_eye_changed = cv.norm(np.array(self.tmp_right_pupil, np.int32), np.array(self.right_pupil, np.int32)) > self.move_thresh
        left_eye_changed = cv.norm(np.array(self.tmp_left_pupil, np.int32), np.array(self.left_pupil, np.int32)) > self.move_thresh

        if right_eye_changed and left_eye_changed:
            self.right_pupil = self.tmp_right_pupil
            self.left_pupil = self.tmp_left_pupil

    def get_images(self):
        # Returns references to all images responsible for GUI
        images = {
            "face_frame": self.face_frame,
            "left_eye_frame": self.left_eye_frame,
            "right_eye_frame": self.right_eye_frame,
            "lp_thresh_frame": self.lp_thresh_frame,
            "rp_thresh_frame": self.rp_thresh_frame,
            "lp_frame": self.lp_frame,
            "rp_frame": self.rp_frame
        }

        return images


class Homography:
    """
        This class represents an instance of Homography
        Homography is a planar relationship that transforms points from one plane to another.
        From a mathematics point of view it is a 3x3 matrix transforming 3 dimensional vectors
        (called homogeneous coordinates) that represent the 2D points on the plane.
        We will use this to track eye movement relative to cursor on screen
    """

    def __init__(self, move_threshold=MOVE_THRESHOLD):
        # Calibrated Eye & Cursor Tracking vectors
        self.calibration_circle_pos = np.empty((0, 2), np.float32)
        self.calibration_eye_pos = np.empty((0, 2), np.float32)

        # Homography & Counter Variables
        self.homography = None
        self.calibration_counter = 0
        self.current_screen_point = [0, 0]

        # Defined threshold ratio
        self.move_threshold = move_threshold

    def getMiddlePoint(self, eyes):
        # Generates approximate center of pupil from eye coordinates
        left_eye = eyes[0]
        right_eye = eyes[1]
        return [(left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2]

    def saveCalibrationPosition(self, eyes_pos, position, pos_counter):
        # We use this function to save calibrations mapping
        # eye movement to cursor movement
        eyes_point = self.getMiddlePoint(eyes_pos)
        eyes_point = [[list(eyes_point)[0], list(eyes_point)[1]]]
        c_pos = [[list(position)[0], list(position)[1]]]

        if self.calibration_counter < pos_counter:
            self.calibration_counter += 1
            self.calibration_circle_pos = np.append(self.calibration_circle_pos, np.array(c_pos), axis=0)
            self.calibration_eye_pos = np.append(self.calibration_eye_pos, np.array(eyes_point), axis=0)

    def calculateHomography(self):
        # findhomography helps in transforming an image that is 
        # present in the form of a three by three matrix and Maps the specific 
        # points present in that image to the corresponding points that are 
        # present in another image that has been provided.
        # Hence we map current eye position relative to its previous position
        self.homography, _ = cv.findHomography(self.calibration_eye_pos, self.calibration_circle_pos)

    def getCursorPos(self, eyes_pos):
        # Fetches cursor position relative to eyes
        eyes_point = np.empty((0, 2), np.float32)
        eyes_point = np.append(eyes_point, np.array([self.getMiddlePoint(eyes_pos)]), axis=0)
        eyes_point = np.append(eyes_point, 1)
        eyes_point_homogenous = np.dot(self.homography, eyes_point)
        screen_point = [eyes_point_homogenous[0] / eyes_point_homogenous[2], eyes_point_homogenous[1] / eyes_point_homogenous[2]]

        # If the newly normalized movement speed is less than threshold, then we ignore changes
        if cv.norm(np.array(screen_point, np.int32), np.array(self.current_screen_point, np.int32)) <= self.move_threshold:
            screen_point = self.current_screen_point

        self.current_screen_point = screen_point
        return screen_point


class GraphicalInterface:
    """
        Creates an Instance of Graphical Interface
        This is where all GUI Rendering & Key Actions are handled
    """

    def __init__(self):
        self.screensize = WINDOW_WIDTH, WINDOW_HEIGHT

        # Initializing Canvas Background
        img = np.random.randint(222, size=(self.screensize[1], self.screensize[0], 3))

        # Canvas Properties
        self.canvas = np.array(img, dtype=np.uint8)
        self.canvas_tmp = np.array(img, dtype=np.uint8)
        self.canvas_w = self.canvas.shape[1] - int(self.screensize[0] * 0.2)
        self.canvas_h = self.canvas.shape[0]

        # UI Element configuration
        self.phase = 0
        self.eye_radius = int(0.025 * self.canvas_w)
        self.cursor_radius = 10
        self.cursor_color = (0, 0, 0)
        self.last_cursor = [-1, -1]

        # States
        self.waiting = False
        self.save_pos = False
        self.drawing_mode = False

        # Canvas Draw Elements
        self.wait_count = 0
        self.step_w = int(0.025 * self.canvas_w)
        self.step_h = int(0.025 * self.canvas_h)
        self.calibration_cursor_color = (0, 0, 255)
        self.calibration_cursor_pos = (self.eye_radius, int(0.025 * self.canvas_h))
        self.last_calibration_checkpoint = -1
        self.calibration_counter = 0
        self.calibration_poses = [
            (self.step_w, self.step_h), (20 * self.step_w, self.step_h), (39 * self.step_w, self.step_h),
            (self.step_w, 20 * self.step_h), (20 * self.step_w, 20 * self.step_h), (39 * self.step_w, 20 * self.step_h),
            (self.step_w, 39 * self.step_h), (20 * self.step_w, 39 * self.step_h), (39 * self.step_w, 39 * self.step_h),
            (10 * self.step_w, self.step_h), (30 * self.step_w, self.step_h),
            (10 * self.step_w, 20 * self.step_h), (30 * self.step_w, 20 * self.step_h),
            (10 * self.step_w, 39 * self.step_h), (30 * self.step_w, 39 * self.step_h),
            (self.step_w, 30 * self.step_h), (39 * self.step_w, 10 * self.step_h),
        ]
        self.offset_y = (self.step_w - self.step_h) if self.step_w > self.step_h else (self.step_h - self.step_w)

    def makeWindow(self, main_image, lateral_images, cursor=None, sensibility=0.95):
        # Renders GUI Content on Window based on Phase
        # Image To Screen Ratio
        ratio = main_image.shape[1] / main_image.shape[0]

        img = np.random.randint(222, size=(self.screensize[1], self.screensize[0], 3))
        img = np.array(img, dtype=np.uint8)

        main_height = int(self.screensize[1] * 0.8)
        main_width = int(main_height * ratio)

        # Main image
        if self.phase == 0:
            main_y_offset = int((self.screensize[1] - main_height) / 3)
            main_x_offset = int((self.screensize[0] - main_width) / 4)
            main_image = cv.resize(main_image, (main_width, main_height))

            img[main_y_offset:main_image.shape[0] + main_y_offset, main_x_offset:main_image.shape[1] + main_x_offset] = main_image

            # Instruction
            img[0:main_y_offset, main_x_offset:main_image.shape[1] + main_x_offset] = cv.blur(img[0:main_y_offset, main_x_offset:main_image.shape[1] + main_x_offset], (10, 10))
            img = cv.putText(img, 'Adjust the threshold, then press space to calibrate [i for info]', (main_x_offset + 10, int(main_y_offset / 2)), cv.FONT_HERSHEY_SIMPLEX, 0.8, color=(255, 255, 255))

        else:
            img = self.canvas

            if self.phase == 2 and not self.drawing_mode:
                img = cv.copyTo(self.canvas_tmp, None)

        # Lateral Bar
        lateral_width = int(self.screensize[0] * 0.2)
        lateral_height = self.screensize[1]
        img[0:img.shape[0], img.shape[1] - lateral_width:img.shape[1]] = (77, 77, 77)

        # Face Zoom Image
        face_frame = lateral_images["face_frame"]
        if face_frame is not None:
            im1_width = int(lateral_width * 0.8)
            im1_height = int(im1_width / ratio)
            im1_x_offset = int(lateral_width * 0.1)
            face_frame = cv.resize(face_frame, (im1_width, im1_height))
            img[40:face_frame.shape[0] + 40, img.shape[1] - lateral_width + im1_x_offset: img.shape[1] - lateral_width + im1_x_offset + im1_width] = face_frame
            img = cv.putText(img, 'Face', (img.shape[1] - lateral_width + int(lateral_width / 2) - 25, 35), cv.FONT_HERSHEY_SIMPLEX, 0.75, color=(242, 242, 242))

        if self.phase == 0:
            left_eye_frame = lateral_images["left_eye_frame"]
            right_eye_frame = lateral_images["right_eye_frame"]

            # Left Eye Image
            if left_eye_frame is not None:
                im2_width = int(lateral_width * 0.3)
                im2_height = im2_width
                im2_x_offset = int(lateral_width * 0.15)
                im2_y_offset = int(lateral_width * 0.45)
                left_eye_frame = cv.resize(left_eye_frame, (im2_width, im2_height))
                img[left_eye_frame.shape[0] + 65 + im2_y_offset:2 * left_eye_frame.shape[0] + 65 + im2_y_offset, img.shape[1] - lateral_width + im2_x_offset: img.shape[1] - lateral_width + im2_x_offset + im2_width] = left_eye_frame

            # Right Eye Image
            if right_eye_frame is not None:
                im3_width = int(lateral_width * 0.3)
                im3_height = im3_width  # int(im3_width / ratio)
                im3_x_offset = int(lateral_width * 0.6)
                im3_y_offset = int(lateral_width * 0.45)
                right_eye_frame = cv.resize(right_eye_frame, (im3_width, im3_height))
                img[right_eye_frame.shape[0] + 65 + im3_y_offset:2 * right_eye_frame.shape[0] + 65 + im3_y_offset, img.shape[1] - lateral_width + im3_x_offset: img.shape[1] - lateral_width + im3_x_offset + im3_width] = right_eye_frame

            if left_eye_frame is not None or right_eye_frame is not None:
                img = cv.putText(img, 'Eyes', (img.shape[1] - lateral_width + int(lateral_width / 2) - 25, face_frame.shape[0] + 105), cv.FONT_HERSHEY_SIMPLEX, 0.75, color=(242, 242, 242))

            # Left Pupil Keypoints Image
            lp_frame = lateral_images["lp_frame"]
            rp_frame = lateral_images["rp_frame"]
            if lp_frame is not None:
                im6_width = int(lateral_width * 0.3)
                im6_height = im6_width  # int(im6_width / ratio)
                im6_x_offset = int(lateral_width * 0.15)
                im6_y_offset = int(lateral_width * 0.45)
                lp_frame = cv.resize(lp_frame, (im6_width, im6_height))
                img[lp_frame.shape[0] + 165 + im6_y_offset:2 * lp_frame.shape[0] + 165 + im6_y_offset,  img.shape[1] - lateral_width + im6_x_offset: img.shape[1] - lateral_width + im6_x_offset + im6_width] = lp_frame

            # Right Pupil Keypoints Image
            if rp_frame is not None:
                im7_width = int(lateral_width * 0.3)
                im7_height = im7_width  # int(im3_width / ratio)
                im7_x_offset = int(lateral_width * 0.6)
                im7_y_offset = int(lateral_width * 0.45)
                rp_frame = cv.resize(rp_frame, (im7_width, im7_height))
                img[rp_frame.shape[0] + 165 + im7_y_offset:2 * rp_frame.shape[0] + 165 + im7_y_offset, img.shape[1] - lateral_width + im7_x_offset: img.shape[1] - lateral_width + im7_x_offset + im7_width] = rp_frame

        elif self.phase == 1:
            img = cv.putText(img, 'Follow the circle!', (img.shape[1] - lateral_width + 50, 2 * lateral_images["right_eye_frame"].shape[0] + 120 + int(lateral_width * 0.45)), cv.FONT_HERSHEY_SIMPLEX, 0.9, color=(252, 252, 252))

        if self.phase > 0:
            # Mode
            mode = 'Paint Mode' if self.drawing_mode else 'Pointer Mode'
            mode = 'Calibration' if self.phase == 1 else mode
            col = (111, 111, 111) if self.drawing_mode else (222, 222, 222)
            col_t = (10, 10, 10) if not self.drawing_mode else (255, 255, 255)
            img[face_frame.shape[0] + 60:face_frame.shape[0] + 100, img.shape[1] - lateral_width: img.shape[1]] = col
            img = cv.putText(img, mode, (img.shape[1] - lateral_width + 30, face_frame.shape[0] + 90), cv.FONT_HERSHEY_DUPLEX, 0.9, color=col_t)

            if cursor is not None and cursor[0] >= 0 and cursor[1] >= 0:
                if not self.drawing_mode:
                    img = cv.circle(img, (int(cursor[0]), int(cursor[1])), self.cursor_radius, self.cursor_color, -1)

                else:
                    if cursor[0] != self.last_cursor[0] and cursor[1] != self.last_cursor[1]:
                        img = cv.circle(img, (int(cursor[0]), int(cursor[1])), self.cursor_radius, self.cursor_color, -1)

                        if self.last_cursor[0] != -1 and self.last_cursor[1] != -1:
                            img = cv.line(img, (int(cursor[0]), int(cursor[1])), (int(self.last_cursor[0]), int(self.last_cursor[1])), self.cursor_color, 2 * self.cursor_radius)

                        self.last_cursor = cursor

        # Sensibility Value
        img = cv.putText(img, 'Sensibility', (img.shape[1] - lateral_width + int(lateral_width / 2) - 55, lateral_height - 200), cv.FONT_HERSHEY_SIMPLEX, 0.75, color=(242, 242, 242))
        img = cv.putText(img, "{:.2f}".format(sensibility), (img.shape[1] - lateral_width + int(lateral_width / 2) - 15, lateral_height - 160), cv.FONT_HERSHEY_SIMPLEX, 0.85, color=(255, 255, 255))
        img = cv.putText(img, 'Press  < to decrease', (img.shape[1] - lateral_width + 10, lateral_height - 120), cv.FONT_HERSHEY_SIMPLEX, 0.5, color=(242, 242, 242))
        img = cv.putText(img, 'and > to increase, press i for info', (img.shape[1] - lateral_width + 10, lateral_height - 100), cv.FONT_HERSHEY_SIMPLEX, 0.5, color=(242, 242, 242))

        if self.phase == 2:
            # Commands
            img = cv.putText(img, 'Commands', (img.shape[1] - lateral_width + 50, face_frame.shape[0] + 130), cv.FONT_HERSHEY_SIMPLEX, 0.9, color=(252, 252, 252))
            img = cv.putText(img, '[SPACE] toggle mode', (img.shape[1] - lateral_width + 20, face_frame.shape[0] + 170), cv.FONT_HERSHEY_SIMPLEX, 0.7, color=(252, 252, 252))
            img = cv.putText(img, '[s] save    [c] clear', (img.shape[1] - lateral_width + 20, face_frame.shape[0] + 200), cv.FONT_HERSHEY_SIMPLEX, 0.7, color=(252, 252, 252))
            img = cv.putText(img, '[+/-] change cursor size', (img.shape[1] - lateral_width + 20, face_frame.shape[0] + 230), cv.FONT_HERSHEY_SIMPLEX, 0.7, color=(252, 252, 252))
            img = cv.putText(img, 'Colors', (img.shape[1] - lateral_width + 80, face_frame.shape[0] + 280), cv.FONT_HERSHEY_SIMPLEX, 0.9, color=(252, 252, 252))
            square_dim = int(lateral_width * 0.1)

            # Test
            letters = ['r', 'g', 'b', 'n', 'w', 'y', 'p', 'a']
            colors = [
                (0, 0, 255),
                (0, 255, 0),
                (255, 0, 0),
                (0, 0, 0),
                (255, 255, 255),
                (0, 255, 255),
                (255, 0, 255),
                (255, 255, 0)
            ]

            for itr in [0, 2, 4, 6]:
                img[face_frame.shape[0] + 300 + int(40 * itr / 2):face_frame.shape[0] + 300 + int(40 * itr / 2) + square_dim, img.shape[1] - lateral_width + 40:img.shape[1] - lateral_width + 40 + square_dim] = colors[itr]
                img[face_frame.shape[0] + 300 + int(40 * itr / 2):face_frame.shape[0] + 300 + int(40 * itr / 2) + square_dim, img.shape[1] - int(lateral_width / 2):img.shape[1] - int(lateral_width / 2) + square_dim] = colors[itr + 1]

                img = cv.putText(img, letters[itr], (img.shape[1] - lateral_width + 50 + square_dim, face_frame.shape[0] + 295 + (int(itr / 2) * 38) + square_dim), cv.FONT_HERSHEY_SIMPLEX, 0.9, color=(252, 252, 252))
                img = cv.putText(img, letters[itr + 1], (img.shape[1] - int(lateral_width / 2) + square_dim + 10, face_frame.shape[0] + 295 + (int(itr / 2) * 38) + square_dim), cv.FONT_HERSHEY_SIMPLEX, 0.9, color=(252, 252, 252))

        cv.imshow(WINDOW_NAME, img)

    def runCalibration(self):
        # Calibrates current coordinates
        self.canvas[:, :] = (255, 255, 255)
        self.wait_count = 0
        self.calibration_cursor_pos = (15 * self.step_w, 15 * self.step_h + self.offset_y)
        self.step_w *= -1
        self.step_h *= -1

    def calibrationStep(self, left_visible=False, right_visible=False):
        # Caliberates a single step
        eyes_visible = left_visible and right_visible
        self.calibration_cursor_color = (0, 255, 0) if eyes_visible else (0, 0, 255)

        if eyes_visible:
            if self.waiting:
                self.wait_count += 1
                self.calibration_cursor_color = (255, 0, 0)

                if self.wait_count == 10:
                    self.wait_count = 0
                    self.waiting = False
                    self.save_pos = False

            else:
                self.calibration_cursor_pos = (
                    list(self.calibration_cursor_pos)[0] + self.step_w,
                    list(self.calibration_cursor_pos)[1] + self.step_h)
                self.checkPosition()

        self.drawCalibrationCanvas()
        self.canvas = cv.circle(self.canvas, self.calibration_cursor_pos, self.eye_radius, self.calibration_cursor_color, -1)
        return self.save_pos

    def checkPosition(self):
        # Checks current position in PHASE 1
        pos_x = int(list(self.calibration_cursor_pos)[0])
        pos_y = int(list(self.calibration_cursor_pos)[1]) - self.offset_y
        pos = (pos_x, pos_y)

        if pos in self.calibration_poses:
            self.save_pos = True if not self.save_pos else False
            self.calibration_cursor_color = (255, 0, 0)
            self.last_calibration_checkpoint += 1

            if not self.waiting:
                self.calibration_counter += 1
                self.waiting = True

            if pos == self.calibration_poses[0]:
                self.step_h = 0
                self.step_w = int(0.025 * self.canvas_w)

            elif pos == self.calibration_poses[2]:
                self.step_h = int(0.025 * self.canvas_h)
                self.step_w = 0

            elif pos == self.calibration_poses[5]:
                self.step_h = 0
                self.step_w = -int(0.025 * self.canvas_w)

            elif pos == self.calibration_poses[3]:
                self.step_h = int(0.025 * self.canvas_h)
                self.step_w = 0

            elif pos == self.calibration_poses[6]:
                self.step_h = 0
                self.step_w = int(0.025 * self.canvas_w)

            elif pos == self.calibration_poses[8]:
                self.step_h = 0
                self.step_w = 0
                self.endCalibration()

    def endCalibration(self):
        # Finishes Calibration
        self.phase = 2
        self.drawing_mode = False
        self.canvas[:, :] = np.array(np.zeros(self.canvas.shape), dtype=np.uint8)
        self.canvas_tmp[:, :] = (255, 255, 255)

    def toggleDrawingMode(self):
        # Enables drawing Mode with Eyes
        self.drawing_mode = not self.drawing_mode
        self.last_cursor = [-1, -1]

        if self.drawing_mode:
            self.canvas = cv.copyTo(self.canvas_tmp, None)

        else:
            self.canvas_tmp = cv.copyTo(self.canvas, None)

    def clearCanvas(self):
        # Cleans Canvas for new drawing
        self.canvas[:, :] = (255, 255, 255)
        self.canvas_tmp[:, :] = (255, 255, 255)

    def changeCursorDimension(self, quantity):
        # Changes Cursor
        self.cursor_radius += quantity

    def alertBox(self, title, message):
        # Outputs a message Box
        messagebox.showinfo(title, message)
        root.update()

    def checkKey(self, key):
        # Listens for user input
        if key == "CLEAR":
            self.clearCanvas()

        elif key.startswith('CURSOR'):
            opts = {
                'CURSOR_INCREASE': 1,
                'CURSOR_DECREASE': -1
            }

            self.changeCursorDimension(opts.get(key))

        elif key == "SCREENSHOT": 
            path = os.path.expanduser("./assets/") + (np.random.randint()).tostring() + ".png"
            cv.imwrite(path, self.canvas)
            self.alertBox("Image saved", "Image saved correctly in " + path)

        elif key.startswith('COLOR'):
            opts = {
                "COLOR_AQUA": (255, 255, 0),
                "COLOR_BLUE": (255, 0, 0),
                "COLOR_GREEN": (0, 255, 0),
                "COLOR_BLACK": (0, 0, 0),
                "COLOR_FUCHSIA  ": (255, 0, 255),
                "COLOR_RED": (0, 0, 255),
                "COLOR_WHITE": (255, 255, 255),
                "COLOR_YELLOW": (0, 255, 255)
            }

            self.cursor_color = opts.get(key)

        else:
            print("Not Listening for this Key")


    def drawCalibrationCanvas(self):
        # Enables calibration run
        self.canvas[:, :] = (255, 255, 255)

        # Draw ghost path
        sp = int(self.cursor_radius)
        checkpoint_poses = [tuple(map(operator.add, e, (sp, sp))) for e in self.calibration_poses]

        self.canvas = cv.line(self.canvas, checkpoint_poses[0], checkpoint_poses[2], (133, 133, 133), self.cursor_radius)
        self.canvas = cv.line(self.canvas, checkpoint_poses[3], checkpoint_poses[5], (133, 133, 133), self.cursor_radius)
        self.canvas = cv.line(self.canvas, checkpoint_poses[6], checkpoint_poses[8], (133, 133, 133), self.cursor_radius)
        self.canvas = cv.line(self.canvas, checkpoint_poses[2], checkpoint_poses[5], (133, 133, 133), self.cursor_radius)
        self.canvas = cv.line(self.canvas, checkpoint_poses[3], checkpoint_poses[6], (133, 133, 133), self.cursor_radius)

        checkpoint_color = (111, 111, 111)
        for checkpoint in self.calibration_poses:
            cv.rectangle(self.canvas, checkpoint, tuple(map(operator.add, checkpoint, (20, 20))), checkpoint_color, -1)

        if self.last_calibration_checkpoint < 0:
            return

        sorted_indices = [0, 9, 1, 10, 2, 16, 5, 12, 4, 11, 3, 15, 6, 13, 7, 14, 8]
        sorted_poses = [self.calibration_poses[idx] for idx in sorted_indices]

        checkpoint_color = (0, 250, 0)
        cv.rectangle(self.canvas, sorted_poses[0], tuple(map(operator.add, sorted_poses[0], (20, 20))), checkpoint_color, -1)

        for square_idx in range(self.last_calibration_checkpoint):
            prev_square = sorted_poses[square_idx]
            square = sorted_poses[square_idx + 1]
            cv.rectangle(self.canvas, prev_square, tuple(map(operator.add, prev_square, (20, 20))), checkpoint_color, -1)
            cv.rectangle(self.canvas, square, tuple(map(operator.add, square, (20, 20))), checkpoint_color, -1)
            self.canvas = cv.line(self.canvas, tuple(map(operator.add, prev_square, (10, 10))), tuple(map(operator.add, square, (10, 10))), checkpoint_color, self.cursor_radius)

        self.canvas = cv.line(self.canvas, tuple(map(operator.add, sorted_poses[self.last_calibration_checkpoint], (10, 10))), tuple(map(operator.add, self.calibration_cursor_pos, (10, 10))), checkpoint_color, self.cursor_radius)


class KeyHandler:
    """
        This class represents an instance of Key Handling
        We use this to handle all user input & window actions
    """

    def __init__(self):
        self.phase = 0
        self.delay = KEY_DELAY

        # Actual KeyStroke
        self.key = None

    def updatePhase(self, ph):
        # Changes Phase for key detection
        self.phase = ph

    def listenForKeyStroke(self):
        # Waits for key stroke
        self.key = cv.waitKey(self.delay)
        return self.getKey()

    def getKey(self):
        keymapping = {
            # Global Keys, will work on all Phases
            27: "EXIT",  # Escape Key
            32: "PHASE",  # Space Key, used to switch phases
            60: "THRESHOLD_DECREASE",  # < Key, used to decrease overlap threshold 
            62: "THRESHOLD_INCREASE",  # > Key, used to increase overlap threshold
            105: "INFO",  # i Key, used to display information box

            # Phase 2 specific, i.e. drawing features
            43: "CURSOR_INCREASE",  # + Key, increases cursor size
            45: "CURSOR_DECREASE",  # - Key, decreases cursor size
            99: "CLEAR",  # c Key, clears the screen
            115: "SCREENSHOT",  # s Key, saves a canvas screenshot

            # Phase 2 specific, colors
            97: "COLOR_AQUA",  # a Key, changes color to aqua
            98: "COLOR_BLUE",  # b Key, changes color to blue
            103: "COLOR_GREEN",  # g Key, changes color to green
            110: "COLOR_BLACK",  # n Key, changes color to black
            112: "COLOR_FUCHSIA",  # p Key, changes color to fuchsia
            114: "COLOR_RED",  # r Key, changes color to red
            119: "COLOR_WHITE",  # w Key, changes color to white
            121: "COLOR_YELLOW",  # y Key, changes color to yellow
        }

        return keymapping.get(self.key, 'NONE')


detector = CascadeDetector()
homo = Homography()
gui = GraphicalInterface()
keys = KeyHandler()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Frame not received, exiting (maybe stream is ending)")
        break

    frame = cv.flip(frame, 1)  # Flipping the frame, i.e L <-> R
    detector.findEyes(frame)  # Pinpoint Eye Coords in given frame

    if PHASE == 1:
        if homo.homography is not None:
            # We have already calibrated for eye-cursor coordination
            PHASE = 2
            detector.startPhase(2)
            gui.endCalibration()

        else:
            # Will continue this step till calibration is successful, via space
            # throws error if eyes not visible
            if gui.calibrationStep(detector.left_is_visible, detector.right_is_visible):
                homo.saveCalibrationPosition([detector.left_pupil, detector.right_pupil], gui.calibration_cursor_pos, gui.calibration_counter)

            if gui.phase == 2:
                PHASE = 2
                homo.calculateHomography()
                detector.startPhase(2)

    elif PHASE == 2:
        # Cursor moved to where eye is currently
        START_CURSOR_POS = homo.getCursorPos([detector.left_pupil, detector.right_pupil])

    gui.makeWindow(frame, detector.get_images(), START_CURSOR_POS, detector.overlap_threshold)

    if cv.getWindowProperty(WINDOW_NAME, 0) == -1:
        # Window is no longer present, hence represents CROSS key
        break

    key = keys.listenForKeyStroke()

    if key == "EXIT":
        break

    elif key.startswith('THRESHOLD'):
        opts = {
            'THRESHOLD_DECREASE': -0.01,
            'THRESHOLD_INCREASE': 0.01
        }

        detector.overlap_threshold += opts.get(key)
    
    elif key == "PHASE":
        if PHASE < 2:
            if PHASE == 0:
                if not detector.left_is_visible or not detector.right_is_visible:
                    gui.alertBox("Error", "Show both your eyes to the camera.")
                    detector.phase -= 1
                    gui.phase -= 1
                    PHASE -= 1

                else:
                    gui.alertBox("Calibration Phase", "Keep still your shoulders and follow the circle with "
                                                       "the eyes, moving with your head as more as possible.")
                    cv.destroyWindow("EyePaint")
                    cv.namedWindow('EyePaint', cv.WINDOW_FULLSCREEN)
                    gui.runCalibration()

            else:
                gui.alertBox("Paint Phase", "Keep still your shoulders and move the cursor with your eyes, "
                                             "changing between drawing/pointing mode with space key. "
                                             "Personalize the cursor and change the color by pressing the "
                                             "relative key on the lateral bar.")

            detector.phase += 1
            gui.phase += 1
            PHASE += 1

        else:
            gui.toggleDrawingMode()

    elif key == "INFO":
        if PHASE == 0:
            gui.alertBox("Info - Sensibility",
                      "Set the eyes detector sensibility: stop when the purple squares around the eyes are "
                      "stable but also they keep following the eyes smoothly.")

        elif PHASE == 1:
            gui.alertBox("Info - Sensibility",
                      "Set the eyes detector sensibility: stop when the purple squares around the eyes are "
                      "stable but also they keep following the eyes smoothly.")

    else:
        gui.checkKey(key)


cap.release()
cv.destroyAllWindows()
