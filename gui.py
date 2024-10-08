import copy
import csv
import itertools
import json
import os
import sys
import time
from collections import Counter, deque

import cv2
import emoji
import mediapipe as mp
import numpy as np
from model import KeyPointClassifier, PointHistoryClassifier
from PySide6.QtCore import QObject, Qt, QThread, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPalette, QPixmap
from PySide6.QtWidgets import (QApplication, QComboBox, QDockWidget,
                               QFileDialog, QGridLayout, QHBoxLayout, QLabel,
                               QLineEdit, QMainWindow, QPushButton,
                               QScrollArea, QSizePolicy, QStackedWidget,
                               QVBoxLayout, QWidget)
from utils.cvfpscalc import CvFpsCalc


class FrameProcessor(QObject):
    updateImage = Signal(QImage)
    updateInfo = Signal(str)

    def __init__(self):
        super().__init__()
        self.cap = cv2.VideoCapture(0)
        self.running = True

        # Load MediaPipe hands model
        self.mp_hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
        )

        # Load classifiers
        self.keypoint_classifier = KeyPointClassifier()
        self.point_history_classifier = PointHistoryClassifier()

        # Read labels ###########################################################
        with open(
            "model/keypoint_classifier/keypoint_classifier_label.csv",
            encoding="utf-8-sig",
        ) as f:
            self.keypoint_classifier_labels = csv.reader(f)
            self.keypoint_classifier_labels = [
                row[0] for row in self.keypoint_classifier_labels
            ]
        with open(
            "model/point_history_classifier/point_history_classifier_label.csv",
            encoding="utf-8-sig",
        ) as f:
            self.point_history_classifier_labels = csv.reader(f)
            self.point_history_classifier_labels = [
                row[0] for row in self.point_history_classifier_labels
            ]

        # Gesture history
        self.history_length = 16
        self.point_history = deque(maxlen=self.history_length)
        self.finger_gesture_history = deque(maxlen=self.history_length)
        self.mode = 0
        self.use_brect = True
        self.gesture_cooldown = {}
        self.gesture_commands = {
        '0': 'echo this is first gesture',
        '1': 'echo this is second gesture',
        '2': 'echo this is third gesture',
        '3': 'echo this is fourth gesture',
        '4': 'echo this is fifth gesture',
        '5': 'echo this is sixth gesture',
        '6': 'echo this is seventh gesture',
        '7': 'echo this is eighth gesture',
        '8': 'echo this is ninth gesture',
            # Add more gestures and their associated system commands
            }
        self.cooldown = 5  # cooldown in seconds


    def run(self):
        while self.running and self.cap.isOpened():
            fps = CvFpsCalc().get()
            key = cv2.waitKey(10)
            number, mode = self.select_mode(key, self.mode)
            ret, image = self.cap.read()
            if not ret:
                break
            image = cv2.flip(image, 1)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            debug_image = copy.deepcopy(image)


            image.flags.writeable = False
            results = self.mp_hands.process(image)
            image.flags.writeable = True

            if results.multi_hand_landmarks is not None:
                for hand_landmarks, handedness in zip(
                    results.multi_hand_landmarks, results.multi_handedness
                ):

                    brect = self.calc_bounding_rect(debug_image, hand_landmarks)
                    landmark_list = self.calc_landmark_list(debug_image, hand_landmarks)

                    pre_processed_landmark_list = self.pre_process_landmark(
                        landmark_list
                    )
                    pre_processed_point_history_list = self.pre_process_point_history(
                        debug_image, self.point_history
                    )
                    self.logging_csv(
                        number,
                        mode,
                        pre_processed_landmark_list,
                        pre_processed_point_history_list,
                    )

                    hand_sign_id = self.keypoint_classifier(pre_processed_landmark_list)
                    if hand_sign_id == 2:
                        # self.point_history.append(landmark_list[8])
                        self.point_history.append([0, 0])
                    else:
                        self.point_history.append([0, 0])
                    hand_sign_id_str = str(hand_sign_id)  # Convert hand_sign_id to string if your keys in gesture_commands are strings

                    if hand_sign_id_str in self.gesture_commands:
                        current_time = time.time()
                        if hand_sign_id_str not in self.gesture_cooldown or current_time - self.gesture_cooldown[hand_sign_id_str] > self.cooldown:
                            os.system(self.gesture_commands[hand_sign_id_str])
                            self.gesture_cooldown[hand_sign_id_str] = current_time


                    finger_gesture_id = 0
                    point_history_len = len(pre_processed_point_history_list)
                    if point_history_len == (self.history_length * 2):
                        finger_gesture_id = self.point_history_classifier(
                            pre_processed_point_history_list
                        )

                    self.finger_gesture_history.append(finger_gesture_id)
                    most_common_fg_id = Counter(self.finger_gesture_history).most_common()

                    debug_image = self.draw_bounding_rect(self.use_brect, debug_image, brect)
                    debug_image = self.draw_landmarks(debug_image, landmark_list)
                    debug_image = self.draw_info_text(debug_image, brect, handedness, self.keypoint_classifier_labels[hand_sign_id],self.point_history_classifier_labels[most_common_fg_id[0][0]],)
                else:
                    self.point_history.append([0,0])

                # debug_image = self.draw_point_history(debug_image, self.point_history)
                debug_image = self.draw_info(debug_image, fps, mode, number)

            h, w, ch = debug_image.shape
            bytes_per_line = ch * w
            qImg = QImage(debug_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.updateImage.emit(qImg)


    def select_mode(self, key, mode):
        number = -1
        if 48 <= key <= 57:  # 0 ~ 9
            number = key - 48
        if key == 110:  # n
            mode = 0
        if key == 107:  # k
            mode = 1
        if key == 104:  # h
            mode = 2
        return number, mode

    def calc_bounding_rect(self, image, landmarks):
        image_width, image_height = image.shape[1], image.shape[0]

        landmark_array = np.empty((0, 2), int)

        for _, landmark in enumerate(landmarks.landmark):
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)

            landmark_point = [np.array((landmark_x, landmark_y))]

            landmark_array = np.append(landmark_array, landmark_point, axis=0)

        x, y, w, h = cv2.boundingRect(landmark_array)

        return [x, y, x + w, y + h]

    def calc_landmark_list(self, image, landmarks):
        image_width, image_height = image.shape[1], image.shape[0]

        landmark_point = []

        # Keypoint
        for _, landmark in enumerate(landmarks.landmark):
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)
            # landmark_z = landmark.z

            landmark_point.append([landmark_x, landmark_y])

        return landmark_point

    def pre_process_landmark(self, landmark_list):
        temp_landmark_list = copy.deepcopy(landmark_list)

        # Convert to relative coordinates
        base_x, base_y = 0, 0
        for index, landmark_point in enumerate(temp_landmark_list):
            if index == 0:
                base_x, base_y = landmark_point[0], landmark_point[1]

            temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
            temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

        # Convert to a one-dimensional list
        temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))

        # Normalization
        max_value = max(list(map(abs, temp_landmark_list)))

        def normalize_(n):
            return n / max_value

        temp_landmark_list = list(map(normalize_, temp_landmark_list))

        return temp_landmark_list

    def pre_process_point_history(self, image, point_history):
        image_width, image_height = image.shape[1], image.shape[0]

        temp_point_history = copy.deepcopy(point_history)

        # Convert to relative coordinates
        base_x, base_y = 0, 0
        for index, point in enumerate(temp_point_history):
            if index == 0:
                base_x, base_y = point[0], point[1]

            temp_point_history[index][0] = (
                temp_point_history[index][0] - base_x
            ) / image_width
            temp_point_history[index][1] = (
                temp_point_history[index][1] - base_y
            ) / image_height

        # Convert to a one-dimensional list
        temp_point_history = list(itertools.chain.from_iterable(temp_point_history))

        return temp_point_history

    def logging_csv(self, number, mode, landmark_list, point_history_list):
        if mode == 0:
            pass
        if mode == 1 and (0 <= number <= 9):
            csv_path = "model/keypoint_classifier/keypoint.csv"
            with open(csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([number, *landmark_list])
        if mode == 2 and (0 <= number <= 9):
            csv_path = "model/point_history_classifier/point_history.csv"
            with open(csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([number, *point_history_list])
        return

    def draw_landmarks(self, image, landmark_point):
        if len(landmark_point) > 0:
            # Thumb
            cv2.line(
                image, tuple(landmark_point[2]), tuple(landmark_point[3]), (0, 0, 0), 6
            )
            cv2.line(
                image,
                tuple(landmark_point[2]),
                tuple(landmark_point[3]),
                (255, 255, 255),
                2,
            )
            cv2.line(
                image, tuple(landmark_point[3]), tuple(landmark_point[4]), (0, 0, 0), 6
            )
            cv2.line(
                image,
                tuple(landmark_point[3]),
                tuple(landmark_point[4]),
                (255, 255, 255),
                2,
            )

            # Index finger
            cv2.line(
                image, tuple(landmark_point[5]), tuple(landmark_point[6]), (0, 0, 0), 6
            )
            cv2.line(
                image,
                tuple(landmark_point[5]),
                tuple(landmark_point[6]),
                (255, 255, 255),
                2,
            )
            cv2.line(
                image, tuple(landmark_point[6]), tuple(landmark_point[7]), (0, 0, 0), 6
            )
            cv2.line(
                image,
                tuple(landmark_point[6]),
                tuple(landmark_point[7]),
                (255, 255, 255),
                2,
            )
            cv2.line(
                image, tuple(landmark_point[7]), tuple(landmark_point[8]), (0, 0, 0), 6
            )
            cv2.line(
                image,
                tuple(landmark_point[7]),
                tuple(landmark_point[8]),
                (255, 255, 255),
                2,
            )

            # Middle finger
            cv2.line(
                image, tuple(landmark_point[9]), tuple(landmark_point[10]), (0, 0, 0), 6
            )
            cv2.line(
                image,
                tuple(landmark_point[9]),
                tuple(landmark_point[10]),
                (255, 255, 255),
                2,
            )
            cv2.line(
                image,
                tuple(landmark_point[10]),
                tuple(landmark_point[11]),
                (0, 0, 0),
                6,
            )
            cv2.line(
                image,
                tuple(landmark_point[10]),
                tuple(landmark_point[11]),
                (255, 255, 255),
                2,
            )
            cv2.line(
                image,
                tuple(landmark_point[11]),
                tuple(landmark_point[12]),
                (0, 0, 0),
                6,
            )
            cv2.line(
                image,
                tuple(landmark_point[11]),
                tuple(landmark_point[12]),
                (255, 255, 255),
                2,
            )

            # Ring finger
            cv2.line(
                image,
                tuple(landmark_point[13]),
                tuple(landmark_point[14]),
                (0, 0, 0),
                6,
            )
            cv2.line(
                image,
                tuple(landmark_point[13]),
                tuple(landmark_point[14]),
                (255, 255, 255),
                2,
            )
            cv2.line(
                image,
                tuple(landmark_point[14]),
                tuple(landmark_point[15]),
                (0, 0, 0),
                6,
            )
            cv2.line(
                image,
                tuple(landmark_point[14]),
                tuple(landmark_point[15]),
                (255, 255, 255),
                2,
            )
            cv2.line(
                image,
                tuple(landmark_point[15]),
                tuple(landmark_point[16]),
                (0, 0, 0),
                6,
            )
            cv2.line(
                image,
                tuple(landmark_point[15]),
                tuple(landmark_point[16]),
                (255, 255, 255),
                2,
            )

            # Little finger
            cv2.line(
                image,
                tuple(landmark_point[17]),
                tuple(landmark_point[18]),
                (0, 0, 0),
                6,
            )
            cv2.line(
                image,
                tuple(landmark_point[17]),
                tuple(landmark_point[18]),
                (255, 255, 255),
                2,
            )
            cv2.line(
                image,
                tuple(landmark_point[18]),
                tuple(landmark_point[19]),
                (0, 0, 0),
                6,
            )
            cv2.line(
                image,
                tuple(landmark_point[18]),
                tuple(landmark_point[19]),
                (255, 255, 255),
                2,
            )
            cv2.line(
                image,
                tuple(landmark_point[19]),
                tuple(landmark_point[20]),
                (0, 0, 0),
                6,
            )
            cv2.line(
                image,
                tuple(landmark_point[19]),
                tuple(landmark_point[20]),
                (255, 255, 255),
                2,
            )

            # Palm
            cv2.line(
                image, tuple(landmark_point[0]), tuple(landmark_point[1]), (0, 0, 0), 6
            )
            cv2.line(
                image,
                tuple(landmark_point[0]),
                tuple(landmark_point[1]),
                (255, 255, 255),
                2,
            )
            cv2.line(
                image, tuple(landmark_point[1]), tuple(landmark_point[2]), (0, 0, 0), 6
            )
            cv2.line(
                image,
                tuple(landmark_point[1]),
                tuple(landmark_point[2]),
                (255, 255, 255),
                2,
            )
            cv2.line(
                image, tuple(landmark_point[2]), tuple(landmark_point[5]), (0, 0, 0), 6
            )
            cv2.line(
                image,
                tuple(landmark_point[2]),
                tuple(landmark_point[5]),
                (255, 255, 255),
                2,
            )
            cv2.line(
                image, tuple(landmark_point[5]), tuple(landmark_point[9]), (0, 0, 0), 6
            )
            cv2.line(
                image,
                tuple(landmark_point[5]),
                tuple(landmark_point[9]),
                (255, 255, 255),
                2,
            )
            cv2.line(
                image, tuple(landmark_point[9]), tuple(landmark_point[13]), (0, 0, 0), 6
            )
            cv2.line(
                image,
                tuple(landmark_point[9]),
                tuple(landmark_point[13]),
                (255, 255, 255),
                2,
            )
            cv2.line(
                image,
                tuple(landmark_point[13]),
                tuple(landmark_point[17]),
                (0, 0, 0),
                6,
            )
            cv2.line(
                image,
                tuple(landmark_point[13]),
                tuple(landmark_point[17]),
                (255, 255, 255),
                2,
            )
            cv2.line(
                image, tuple(landmark_point[17]), tuple(landmark_point[0]), (0, 0, 0), 6
            )
            cv2.line(
                image,
                tuple(landmark_point[17]),
                tuple(landmark_point[0]),
                (255, 255, 255),
                2,
            )

        # Key Points
        for index, landmark in enumerate(landmark_point):
            if index == 0:  # 手首1
                cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
                cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 1:  # 手首2
                cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
                cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 2:  # 親指：付け根
                cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
                cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 3:  # 親指：第1関節
                cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
                cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 4:  # 親指：指先
                cv2.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255), -1)
                cv2.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
            if index == 5:  # 人差指：付け根
                cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
                cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 6:  # 人差指：第2関節
                cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
                cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 7:  # 人差指：第1関節
                cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
                cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 8:  # 人差指：指先
                cv2.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255), -1)
                cv2.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
            if index == 9:  # 中指：付け根
                cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
                cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 10:  # 中指：第2関節
                cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
                cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 11:  # 中指：第1関節
                cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
                cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 12:  # 中指：指先
                cv2.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255), -1)
                cv2.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
            if index == 13:  # 薬指：付け根
                cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
                cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 14:  # 薬指：第2関節
                cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
                cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 15:  # 薬指：第1関節
                cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
                cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 16:  # 薬指：指先
                cv2.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255), -1)
                cv2.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
            if index == 17:  # 小指：付け根
                cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
                cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 18:  # 小指：第2関節
                cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
                cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 19:  # 小指：第1関節
                cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
                cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 20:  # 小指：指先
                cv2.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255), -1)
                cv2.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)

        return image

    def draw_bounding_rect(self, use_brect, image, brect):
        if use_brect:
            # Outer rectangle
            cv2.rectangle(
                image, (brect[0], brect[1]), (brect[2], brect[3]), (0, 0, 0), 1
            )

        return image

    def draw_info_text(
        self, image, brect, handedness, hand_sign_text, finger_gesture_text
    ):
        cv2.rectangle(
            image, (brect[0], brect[1]), (brect[2], brect[1] - 22), (0, 0, 0), -1
        )

        info_text = handedness.classification[0].label[0:]
        if hand_sign_text != "":
            info_text = info_text + ":" + hand_sign_text
        cv2.putText(
            image,
            info_text,
            (brect[0] + 5, brect[1] - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        if finger_gesture_text != "":
            cv2.putText(
                image,
                "Finger Gesture:" + finger_gesture_text,
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 0),
                4,
                cv2.LINE_AA,
            )
            cv2.putText(
                image,
                "Finger Gesture:" + finger_gesture_text,
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        return image

    def draw_point_history(self, image, point_history):
        for index, point in enumerate(point_history):
            if point[0] != 0 and point[1] != 0:
                cv2.circle(
                    image, (point[0], point[1]), 1 + int(index / 2), (152, 251, 152), 2
                )

        return image

    def draw_info(self, image, fps, mode, number):
        cv2.putText(
            image,
            "FPS:" + str(fps),
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            "FPS:" + str(fps),
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        mode_string = ["Logging Key Point", "Logging Point History"]
        if 1 <= mode <= 2:
            cv2.putText(
                image,
                "MODE:" + mode_string[mode - 1],
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            if 0 <= number <= 9:
                cv2.putText(
                    image,
                    "NUM:" + str(number),
                    (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
        return image

    def stop(self):
        self.running = False
        self.cap.release()

class SidebarWidget(QWidget):
    switch_window = Signal(int)

    def __init__(self, stacked_widget):
        super().__init__()
        layout = QVBoxLayout(self)

        # Modern Sidebar Styling
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(QPalette.Window, QColor(28, 30, 38))  # Sleek dark color
        self.setPalette(palette)

        # Sidebar Custom Buttons
        self.buttons = []
        for index, label in enumerate(["Commands", "Preview"]):
            button = QPushButton(label)
            button.clicked.connect(self.create_click_handler(index))
            layout.addWidget(button)
            self.buttons.append(button)
            button.setStyleSheet("""
                QPushButton {
                    background-color: #2C2F3C;
                    color: #FFFFFF;
                    border: 1px solid #3C3F4C;
                    padding: 15px;
                    text-align: left; 
                    font-family: 'Segoe UI';
                    font-size: 14px;
                }
                QPushButton:hover {
                    background-color: #56b6c2; /* Hover effect */
                }
            """)
            button.setMinimumHeight(50)

        layout.addStretch()

    def create_click_handler(self, index):
        def handler():
            self.switch_window.emit(index)
            self.update_button_styles(index)
        return handler
    
    def update_button_styles(self, active_index):
        for i, button in enumerate(self.buttons):
            if i == active_index:
                button.setStyleSheet(button.styleSheet() + "border-left: 4px solid #56b6c2;")
            else:
                button.setStyleSheet(button.styleSheet().replace("border-left: 4px solid #56b6c2;", ""))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GestuNova")
        self.setFixedSize(1200, 900)
        self.setStyleSheet("background-color: #2c2f3c;")

        # Central Widget and Layout
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Stacked Widget for Content (Main Window)
        self.stacked_widget = QStackedWidget()
        main_layout.addWidget(self.stacked_widget)

        # Sidebar Widget
        self.sidebar_dock = QDockWidget(self)
        sidebar_widget = SidebarWidget(self.stacked_widget)
        sidebar_widget.switch_window.connect(self.stacked_widget.setCurrentIndex)
        self.sidebar_dock.setWidget(sidebar_widget)
        self.sidebar_dock.setFeatures(QDockWidget.NoDockWidgetFeatures)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.sidebar_dock)

        # Pages
        self.create_command_binding_page()
        self.create_preview_page()

    def create_command_binding_page(self):
        command_binding_widget = QWidget()
        command_binding_layout = QVBoxLayout(command_binding_widget)

        # Command Binding UI Elements
        grid = QGridLayout()
        command_binding_layout.addLayout(grid)
        command_title_label = QLabel("<h3 style='color: #FFFFFF;'><b><i>Command Binding</i></b></h3>")
        grid.addWidget(command_title_label, 0, 0, 1, 3)
        command_title_label.setMaximumHeight(30)

        self.text_entries = {}
        self.emoji_mapping = {
            "open": emoji.emojize(":hand_with_fingers_splayed:"),  
            "close": emoji.emojize(":raised_fist:"), 
            "pointer": emoji.emojize(":backhand_index_pointing_right:"),
            "okay": emoji.emojize(":OK_hand:"), 
            "peace": emoji.emojize(":victory_hand:"), 
            "call": emoji.emojize(":call_me_hand:"),
            "thumbs up": emoji.emojize(":thumbs_up:"), 
            "thumbs down": emoji.emojize(":thumbs_down:"),
            "rock": emoji.emojize(":love-you_gesture:")
        }

        for i, label in enumerate(self.emoji_mapping):
            emoji_label = self.emoji_mapping[label]
            label_widget = QLabel(f"<h3 style='color: #FFFFFF;'>{emoji_label}</h3>")
            textbox = QLineEdit()
            font = QFont("Noto Color Emoji", 30)  # Use the Noto Color Emoji font
            label_widget.setFont(font)
            textbox.setPlaceholderText("Command")
            textbox.setToolTip("Enter the command to execute for this gesture. Hint: Can concatenate multiple commands with '&&'")
            textbox.setStyleSheet("background-color: #2C2F3C; color: #FFFFFF; padding: 5px; border: 1px solid #3C3F4C;")
            grid.addWidget(label_widget, i % 3 + 1, i // 3 * 2)
            grid.addWidget(textbox, i % 3 + 1, i // 3 * 2 + 1)
            self.text_entries[label] = textbox

        # Save, Export, and Import buttons
        save_button = QPushButton("Save")
        export_button = QPushButton("Export")
        import_button = QPushButton("Import")

        for button in (save_button, export_button, import_button):
            button.setStyleSheet("""
                QPushButton {
                    background-color: #56b6c2;
                    color: #FFFFFF;
                    border: none;
                    padding: 10px;
                    font-family: 'Segoe UI';
                    font-size: 12px;
                }
                QPushButton:hover {
                    background-color: #65c7d6;
                }
            """)
            button.setMinimumHeight(40)

        grid.addWidget(save_button, len(self.emoji_mapping) // 3 + 2, 0)
        grid.addWidget(export_button, len(self.emoji_mapping) // 3 + 2, 1)
        grid.addWidget(import_button, len(self.emoji_mapping) // 3 + 2, 2)

        save_button.clicked.connect(self.update_gesture_commands)
        export_button.clicked.connect(self.on_export_clicked)
        import_button.clicked.connect(self.on_import_clicked)

        self.stacked_widget.addWidget(command_binding_widget)

    def create_preview_page(self):
        preview_widget = QWidget()
        preview_layout = QVBoxLayout(preview_widget)

        # Video Preview (with larger size)
        video_wrapper = QWidget()
        video_wrapper_layout = QHBoxLayout(video_wrapper)
        video_wrapper_layout.setAlignment(Qt.AlignCenter)
        self.video_label = QLabel()  # No initial text, just an empty label
        self.video_label.setStyleSheet(
            "padding: 0px; border: 2px solid #999; background-color: #000000;"
        )
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        width = 640  # Set a fixed width for the video
        height = 480  # Maintain aspect ratio
        self.video_label.setFixedSize(width, height)
        video_wrapper_layout.addWidget(self.video_label)
        preview_layout.addWidget(video_wrapper)

        # Initialize the frame processor and setup the thread for video processing
        self.processor = FrameProcessor()
        self.thread = QThread()
        self.processor.moveToThread(self.thread)
        self.processor.updateImage.connect(self.update_image)
        self.thread.started.connect(self.processor.run)
        self.thread.start()

        controls_row = QHBoxLayout()
        self.cooldown_textbox = QLineEdit()

        # Cooldown textbox
        self.cooldown_textbox.setPlaceholderText("Gesture Cooldown (s)")
        self.cooldown_textbox.setToolTip("Set the cooldown time between executing the same gesture")
        self.cooldown_textbox.setStyleSheet("background-color: #2C2F3C; color: #FFFFFF; padding: 5px; border: 1px solid #3C3F4C;")
        controls_row.addWidget(self.cooldown_textbox)

        save_cooldown_btn = QPushButton("Save")
        save_cooldown_btn.setToolTip("Save the cooldown time")
        save_cooldown_btn.setStyleSheet("""
            QPushButton {
                background-color: #56b6c2;
                color: #FFFFFF;
                border: none;
                padding: 10px;
                font-family: 'Segoe UI';
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #65c7d6;
            }
        """)
        controls_row.addWidget(save_cooldown_btn)

        preview_layout.addLayout(controls_row)

        self.stacked_widget.addWidget(preview_widget)

    def update_image(self, qImg):
        self.video_label.setPixmap(QPixmap.fromImage(qImg))

    def on_export_clicked(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Command Configurations", "", "JSON Files (*.json)", options=options)
        if file_path:
            self.export_commands(file_path)
            print(f"Commands exported to {file_path}")

    def on_import_clicked(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self,"Open Command Configurations", "", "JSON Files (*.json)", options=options)
        if file_path:
            self.import_commands(file_path)
            print(f"Commands imported from {file_path}")

    def export_commands(self, file_path):
        commands = {label: self.text_entries[label].text() for label in self.text_entries}
        with open(file_path, 'w') as file:
            json.dump(commands, file, indent=4)

    def import_commands(self, file_path):
        with open(file_path, 'r') as file:
            commands = json.load(file)
            for label, command in commands.items():
                if label in self.text_entries:
                    self.text_entries[label].setText(command)
            self.processor.gesture_commands = {str(i): command for i, (label, command) in enumerate(commands.items())}

    def update_cooldown(self):
        new_cooldown_value_str = self.cooldown_textbox.text()

        try:
            new_cooldown_value = float(new_cooldown_value_str)
            self.processor.cooldown = new_cooldown_value
        except ValueError:
            print("Invalid cooldown value entered:", new_cooldown_value_str)
            return

        self.cooldown = new_cooldown_value

    def update_gesture_commands(self):
        self.gesture_commands = {}
        for i, label in enumerate(self.emoji_mapping):
            command = self.text_entries[label].text()
            self.processor.gesture_commands[str(i)] = command
        print(self.processor.gesture_commands)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec()
