import cv2 as cv
import time
import HandTrackingModule as htm
import math
import numpy as np
import random
from scipy.spatial.distance import cdist
import pyautogui


# Create the variable of number of Right-Left move
right = 0
left = 0

# Initial Coordinate of Mouse
mouse_x, mouse_y = pyautogui.position()
frame_width1, frame_height1 = 640, 300
screen_width, screen_height = 1920, 1080

detector = htm.HandDetector(maxHands=1)


wCam, hCam = 640, 480

cap = cv.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

HighestPoint = [4, 8, 12, 16, 20]


new_frame_time = 0
prev_frame_time = 0
troList = []
MoveThumb_List = []
previous_point = None
threshold_checkmove = 8.0

# Frame
frame_width = 640
frame_height = 480

def fit_circle(points):
    best_center = None
    best_radius = 0
    best_inliers = 0

    num_iterations = 30
    threshold = 50.0

    for _ in range(num_iterations):
        # Randomly select 3 points
        #print(points)
        sample = random.sample(points, 3)
        x = np.array([p[0] for p in sample])
        y = np.array([p[1] for p in sample])

        # Fit a circle to the selected points
        A = np.column_stack((x, y, np.ones(3)))
        b = -(x**2 + y**2)
        center, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        center_x, center_y = center[:2]
        radius = math.sqrt(center_x**2 + center_y**2 - center[2])

        # Count inliers
        inliers = 0
        for point in points:
            distance = math.sqrt((point[0] - center_x)**2 + (point[1] - center_y)**2)
            if abs(distance - radius) <= threshold:
                inliers += 1

        # Update best circle if we found more inliers
        if inliers > best_inliers:
            best_inliers = inliers
            best_center = (center_x, center_y)
            best_radius = radius

    return best_center, best_radius

#Time drapping algorithm

# def time_dynamic_warping(trajectory1, trajectory2):
#     n, m = len(trajectory1), len(trajectory2)
#
#     # Tính ma trận tương đồng giữa hai chuỗi quỹ đạo
#     similarity_matrix = cdist(trajectory1, trajectory2, metric='euclidean')
#
#     # Tạo ma trận đường đi tối ưu
#     optimal_path_matrix = np.zeros((n, m))
#     optimal_path_matrix[0, 0] = similarity_matrix[0, 0]
#
#     for i in range(1, n):
#         optimal_path_matrix[i, 0] = similarity_matrix[i, 0] + optimal_path_matrix[i-1, 0]
#
#     for j in range(1, m):
#         optimal_path_matrix[0, j] = similarity_matrix[0, j] + optimal_path_matrix[0, j-1]
#
#     for i in range(1, n):
#         for j in range(1, m):
#             optimal_path_matrix[i, j] = similarity_matrix[i, j] + min(optimal_path_matrix[i-1, j-1],
#                                                                      optimal_path_matrix[i-1, j],
#                                                                      optimal_path_matrix[i, j-1])
#
#     # Tính toán độ tương đồng tối ưu giữa hai chuỗi quỹ đạo
#     optimal_similarity = optimal_path_matrix[-1, -1] / (n + m)
#
#     return optimal_similarity



# Check the oscillation of point 4
# Minimum variation to consider oscillation (point 4)
min_variation = 8
# Function to check the oscillation of a list of values

def check_finger_4_oscillation(values):
    if len(values) <= 1:
        return False
    variations = [abs(values[i][0] - values[i-1][0]) for i in range(1, len(values))]
    avg_variation = sum(variations) / len(variations)
    print(avg_variation)
    return avg_variation > min_variation

while True:
    ret, image = cap.read()
    image = detector.findHands(image)
    lmList, xmin, ymin = detector.findPosition(image, handNo=0, draw=True)

    if len(lmList) != 0:
        #print(lmList)
        fingers = []
        cx4, cy4 = int(lmList[4][1]), int(lmList[4][2])
        cx5, cy5 = int(lmList[5][1]), int(lmList[5][2])
        #print(lmList_standard[4][1],lmList_standard[4][2])

        for id in range(0, 5):  #id 0->4
            if id == 0: #Thumb
                if lmList[HighestPoint[id]][1] > lmList[HighestPoint[id]-1][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)
            else:
                if lmList[HighestPoint[id]][2] < lmList[HighestPoint[id]-2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)

        #Open status
        if fingers[0] == 1 and fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 1 and fingers[4] == 1:
            print("Open")
            troList = []

        #check OK_state
        #review Length of Thumb
        if fingers[2] == 1 and fingers[3] == 1 and fingers[4] == 1:
            point4 = int(lmList[4][1]), int(lmList[4][2])
            point8 = int(lmList[8][1]), int(lmList[8][2])

            if abs(detector.cal_distance(point4, point8)) < 50:
                #print(detector.cal_distance(point4, point8))
                print("OK status")


        #Check THUMB move RIGHT-LEFT
        if fingers[0] == 1:
            MoveThumb_condition = True
            for id in range(1, 5):
                if fingers[id] == 1:
                    MoveThumb_condition = False
                    MoveThumb_List = []
                    break
            if MoveThumb_condition:
                Thumb_point = lmList[4][1], lmList[4][2]
                MoveThumb_List.append(Thumb_point)

                if check_finger_4_oscillation(MoveThumb_List):
                    if MoveThumb_List[-1][0] > MoveThumb_List[-2][0]:
                        print("Moving Left")
                        left = left + 1
                    else:
                        print("Moving Right")
                        right = right + 1


                if right > 3:
                    # Press key "Alt" + "Right"
                    pyautogui.hotkey("alt", "right")
                    # Wait seconds
                    time.sleep(3)
                    right = 0
                if left > 3:
                    # Press key "Alt" + "Left"
                    pyautogui.hotkey("alt", "left")
                    # Wait seconds
                    time.sleep(3)
                    left = 0

        #Check circle
        if fingers[0] == 0 and fingers[1] == 1 and fingers[2] == 0 and fingers[3] == 0 and fingers[4] == 0:
            #print("Ngontro")
            newpoint = lmList[8][1], lmList[8][2]
            troList.append(newpoint)
            finger_points = troList

            #if check_finger_4_oscillation(troList):
            if len(troList) > 50:
                center, radius = fit_circle(troList)
                if center is not None:
                    print("Circle found: Center = {}, Radius = {}".format(center, radius))
                    # pyautogui.hotkey('ctrl', '+')  # Thực hiện việc zoom bản đồ lên
                    # time.sleep(5)


        ## Control pointer point
        if fingers[0] == 0 and fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 0 and fingers[4] == 0:
            prev_point8 = None
            newpoint = lmList[8][1], lmList[8][2]
            troList.append(newpoint)
            current_point = newpoint

            first_point = troList[0]
            last_point = troList[-1]

            screen_x = (newpoint[0] / frame_width1) * screen_width

            if newpoint[1] < 100:
                screen_y = ((newpoint[1]-30) / frame_height1) * screen_height
                #print(screen_y)
            else:
                screen_y = ((newpoint[1]) / frame_height1) * screen_height
                #print(screen_y)
            pyautogui.moveTo(screen_x, screen_y, duration=0.1)

            # Update coordinate of mouse
            mouse_x, mouse_y = newpoint[0], newpoint[1]


        #Check CLICK
        if fingers[0] == 1 and fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 0 and fingers[4] == 0:
            x2, y2 = lmList[2][1], lmList[2][2]
            x3, y3 = lmList[3][1], lmList[3][2]
            x5, y5 = lmList[5][1], lmList[5][2]

            point2 = (x2, y2)
            point3 = (x3, y3)
            point5 = (x5, y5)
            angle = detector.calculate_angle(point3, point2, point5)
            cv.line(image, point2, point3, (255,255,0), 3)
            cv.line(image, point2, point5, (255, 255, 0), 3)
            if(abs(angle) > 80):
                print("Click");
                #Press mouse
                pyautogui.mouseDown()
                # Delay
                time.sleep(2)
                # Release mouse
                pyautogui.mouseUp()

        #CHECK PANE

        if fingers[2] == 0 and fingers[3] == 0 and fingers[4] == 0:
            # PANE status
            point4 = int(lmList[4][1]), int(lmList[4][2])
            point8 = int(lmList[8][1]), int(lmList[8][2])

            if point8[1] < lmList[6][2]:
                if abs(detector.cal_distance(point4, point8)) < 20:
                    print("PANE")
                    # Press left mouse
                    pyautogui.mouseDown(button='left')

    else:
        troList =[]
        MoveThumb_List = []


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
    cv.putText(image, fps, (10,80), cv.FONT_ITALIC, 3, (100, 255, 0), 5, cv.LINE_AA)
    cv.circle(image, (10, 80), 10, (0, 255, 255), -1)


    cv.imshow("Frame", image)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()