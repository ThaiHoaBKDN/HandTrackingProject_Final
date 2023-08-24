import cv2 as cv
import time
import mediapipe as mp
import math

class HandDetector():

    def __init__(self, mode=False, maxHands=2, modelComplexity=1, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.modelComplex = modelComplexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mp_drawing_util = mp.solutions.drawing_utils
        # self.mp_drawing_style = mp.solutions.drawing_styles
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplex,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils


    #Member function
    def findHands(self, image, handNo =0, draw=True):
        imageRGB = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        result = self.hands.process(imageRGB) #palm dectection

        if result.multi_hand_landmarks:
            for handlms in result.multi_hand_landmarks:
                if draw:
                    if result.multi_handedness and handNo < len(result.multi_handedness):
                        handedness = result.multi_handedness[handNo].classification[0].label
                        if handedness == "Left":
                            self.mpDraw.draw_landmarks(
                                image,
                                handlms,
                                self.mpHands.HAND_CONNECTIONS,
                                # self.mp_drawing_style.get_default_hand_landmarks_style(),
                                # self.mp_drawing_style.get_default_hand_connections_style()
                            )
        return image

    def findPosition(self, image, handNo=0, draw=True):
        imageRGB = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        result = self.hands.process(imageRGB)
        lmList = []
        lmList_standard = []
        xList = []
        yList = []
        area_box = []
        xmin = []
        ymin = []
        if result.multi_hand_landmarks:
            myHand = result.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                #if result.multi_handedness and handNo < len(result.multi_handedness):
                    handedness = result.multi_handedness[handNo].classification[0].label
                    if handedness == "Left":
                        h, w, _ = image.shape
                        # calculate Position of all points
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        lmList.append([id, cx, cy])
                        lmList_standard.append([id,lm.x,lm.y])

                        xList.append(cx)
                        yList.append(cy)

                        xmin, xmax = min(xList), max(xList)
                        ymin, ymax = min(yList), max(yList)
                        boxW, boxH = xmax - xmin, ymax - ymin
                        bbox = xmin, ymin, boxW, boxH


            if result.multi_handedness and handNo < len(result.multi_handedness):
                handedness = result.multi_handedness[handNo].classification[0].label
                if handedness == "Left":
                    if draw:
                        cv.rectangle(image, (bbox[0] - 20, bbox[1] - 20),
                                     (bbox[0] + bbox[2] + 20, bbox[1] + bbox[3] + 20),
                                     (255, 0, 255), 2)
                    # if id == 8:
                    # cv.circle(image, (cx, cy), 10, (0, 255, 0), cv.FILLED)
        return lmList, xmin, ymin



    def calculate_angle(self, point1, point2, point3):
        # Calculate vector V12
        vector1 = [point1[0] - point2[0], point1[1] - point2[1]]
        # Calcualte vector V23
        vector2 = [point3[0] - point2[0], point3[1] - point2[1]]

        # Calculate length of V12 and V23
        length1 = math.sqrt(vector1[0] ** 2 + vector1[1] ** 2)
        length2 = math.sqrt(vector2[0] ** 2 + vector2[1] ** 2)

        # Dot product of 2 vector
        dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]

        # Calculate angle between 2 vectors
        angle_radians = math.acos(dot_product / (length1 * length2))
        angle_degrees = math.degrees(angle_radians)

        return angle_degrees


    def cal_distance(self, point1, point2):
        distance = math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)
        return distance

    def trackFingerMotion(self,lmList):
        global history, motion_gradient, global_orientation


def main():
    # used to record the time when we processed last frame
    prev_frame_time = 0
    # used to record the time at which we processed current frame
    new_frame_time = 0
    video = cv.VideoCapture(0)
    #detector = HandDetector(detectionCon=0.5)
    detector = HandDetector()
    while (True):
        ret, image = video.read()
        image = detector.findHands(image)
        # time when we finish processing for this frame
        new_frame_time = time.time()
        # Calculating the fps
        # fps will be number of frame processed in given time frame
        # since their will be most of time error of 0.001 second
        # we will be subtracting it to get more accurate result
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        # converting the fps into integer
        fps = int(fps)

        # converting the fps to string so that we can display it on frame
        # by using putText function
        fps = str(fps)
        # putting the FPS count on the frame
        cv.putText(image, fps, (10, 80), cv.FONT_ITALIC, 3, (100, 255, 0), 5, cv.LINE_AA)

        cv.imshow('Hand gesture', image)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    main()
