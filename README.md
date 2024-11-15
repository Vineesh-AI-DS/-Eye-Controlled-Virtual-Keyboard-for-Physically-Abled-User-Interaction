# -Eye-Controlled-Virtual-Keyboard-for-Physically-Abled-User-Interaction
import cv2
import numpy as np
import dlib
from math import hypot
import winsound

# Initialize the face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Set font for keyboard letters
font = cv2.FONT_HERSHEY_PLAIN

# Keyboard setting
keyboard = np.zeros((600, 1000, 3), np.uint8)

# Dictionary containing all letters in the alphabet, each associated with an index
keys_set_1 = {
    0: "Q", 1: "W", 2: "E", 3: "R", 4: "T", 5: "Y", 6: "U", 7: "I", 8: "O", 9: "P",
    10: "A", 11: "S", 12: "D", 13: "F", 14: "G", 15: "H", 16: "J", 17: "K", 18: "L",
    19: "Z", 20: "X", 21: "C", 22: "V", 23: "B", 24: "N", 25: "M"
}

# Function to draw each letter on the virtual keyboard
def letter(letter_index, text, letter_light):
    # Define positions for each letter
    x = (letter_index % 10) * 100
    y = (letter_index // 10) * 200
    width, height = 100, 200
    th = 3  # thickness

    # Draw letter key rectangle
    if letter_light:
        cv2.rectangle(keyboard, (x + th, y + th), (x + width - th, y + height - th), (255, 255, 255), -1)
    else:
        cv2.rectangle(keyboard, (x + th, y + th), (x + width - th, y + height - th), (255, 0, 0), th)

    # Center text inside rectangle
    font_scale, font_th = 4, 2
    text_size = cv2.getTextSize(text, font, font_scale, font_th)[0]
    text_x = x + (width - text_size[0]) // 2
    text_y = y + (height + text_size[1]) // 2
    cv2.putText(keyboard, text, (text_x, text_y), font, font_scale, (255, 0, 0), font_th)

# Helper functions for midpoint and eye blink detection
def midpoint(p1, p2):
    return (p1.x + p2.x) // 2, (p1.y + p2.y) // 2

def get_blinking_ratio(eye_points, facial_landmarks):
    left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
    right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
    center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
    center_bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))
    hor_line_length = hypot(left_point[0] - right_point[0], left_point[1] - right_point[1])
    ver_line_length = hypot(center_top[0] - center_bottom[0], center_top[1] - center_bottom[1])
    return hor_line_length / ver_line_length

def get_gaze_ratio(eye_points, facial_landmarks):
    left_eye_region = np.array([(facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y),
                                (facial_landmarks.part(eye_points[1]).x, facial_landmarks.part(eye_points[1]).y),
                                (facial_landmarks.part(eye_points[2]).x, facial_landmarks.part(eye_points[2]).y),
                                (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y),
                                (facial_landmarks.part(eye_points[4]).x, facial_landmarks.part(eye_points[4]).y),
                                (facial_landmarks.part(eye_points[5]).x, facial_landmarks.part(eye_points[5]).y)], np.int32)

    height, width, _ = frame.shape
    mask = np.zeros((height, width), np.uint8)
    cv2.polylines(mask, [left_eye_region], True, 255, 2)
    cv2.fillPoly(mask, [left_eye_region], 255)
    eye = cv2.bitwise_and(gray, gray, mask=mask)
    
    min_x = np.min(left_eye_region[:, 0])
    max_x = np.max(left_eye_region[:, 0])
    min_y = np.min(left_eye_region[:, 1])
    max_y = np.max(left_eye_region[:, 1])
    gray_eye = eye[min_y: max_y, min_x: max_x]
    
    _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)
    height, width = threshold_eye.shape  # Define width and height for gray_eye (threshold_eye)
    
    left_side_threshold = threshold_eye[0: height, 0: width // 2]
    left_side_white = cv2.countNonZero(left_side_threshold)
    right_side_threshold = threshold_eye[0: height, width // 2: width]
    right_side_white = cv2.countNonZero(right_side_threshold)
    
    if left_side_white == 0:
        gaze_ratio = 1
    elif right_side_white == 0:
        gaze_ratio = 5
    else:
        gaze_ratio = left_side_white / right_side_white
    
    return gaze_ratio

cap = cv2.VideoCapture(0)
board = np.full((500, 500), 255, np.uint8)
frames, blinking_frames, letter_index = 0, 0, 0
keyboard_selected, last_keyboard_selected, text = "left", "left", ""

while True:
    _, frame = cap.read()
    frames += 1
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
    keyboard[:] = (0, 0, 0)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    active_letter = keys_set_1[letter_index]
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)
        left_eye_ratio = get_blinking_ratio([36, 37, 38, 39, 40, 41], landmarks)
        right_eye_ratio = get_blinking_ratio([42, 43, 44, 45, 46, 47], landmarks)
        blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2

        if blinking_ratio > 5.7:
            cv2.putText(frame, "BLINKING", (50, 150), font, 4, (255, 0, 0), thickness=3)
            blinking_frames += 1
            frames -= 1
            if blinking_frames == 5:
                text += active_letter
                winsound.PlaySound("sound.wav", winsound.SND_ALIAS)
        else:
            blinking_frames = 0

        gaze_ratio = (get_gaze_ratio([36, 37, 38, 39, 40, 41], landmarks) + get_gaze_ratio([42, 43, 44, 45, 46, 47], landmarks)) / 2
        if gaze_ratio <= 0.9:
            keyboard_selected = "right"
            if last_keyboard_selected != keyboard_selected:
                winsound.PlaySound("right.wav", winsound.SND_ALIAS)
                last_keyboard_selected = keyboard_selected
        else:
            keyboard_selected = "left"
            if last_keyboard_selected != keyboard_selected:
                winsound.PlaySound("left.wav", winsound.SND_ALIAS)
                last_keyboard_selected = keyboard_selected

    if frames == 15:
        letter_index = (letter_index + 1) % 26
        frames = 0

    for i in range(26):
        letter(i, keys_set_1[i], i == letter_index)

    cv2.putText(board, text, (10, 100), font, 4, 0, 3)
    cv2.imshow("Frame", frame)
    cv2.imshow("Keyboard", keyboard)
    cv2.imshow("Board", board)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
