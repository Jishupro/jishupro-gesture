#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import argparse

import cv2 as cv
import numpy as np
import mediapipe as mp

from utils import CvFpsCalc


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    parser.add_argument("--max_num_hands", type=int, default=2)
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.5)

    parser.add_argument('--use_brect', action='store_true')

    args = parser.parse_args()

    return args


def main():
    
    # 引数解析 #################################################################
    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    max_num_hands = args.max_num_hands
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    use_brect = args.use_brect

    # カメラ準備 ###############################################################
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # モデルロード #############################################################
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=max_num_hands,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    # FPS計測モジュール ########################################################
    cvFpsCalc = CvFpsCalc(buffer_len=10)
    right_count = 0
    threshold = 1
    threshold_right = 4 
    pre_cx = 500
    pre_cy = 500
    right_flg = False
    while True:
        display_fps = cvFpsCalc.get()
        # カメラキャプチャ #####################################################
        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)  # ミラー表示
        debug_image = copy.deepcopy(image)

        # 検出実施 #############################################################
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        results = hands.process(image)
        # 描画 ################################################################
        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                  results.multi_handedness):
                
                # 手の平重心計算
                cx, cy = calc_palm_moment(debug_image, hand_landmarks)
                # print("***cx: ", cx, "***cy: ", cy)
                if cx != pre_cx and abs(cy - pre_cy) / abs(cx - pre_cx) < threshold:
                    if 0 < cx - pre_cx:
                        print("migi")
                        right_count += 1 
                        if threshold_right < right_count and not right_flg:
                            right_flg = True
                            print("******************************")
                            right_count = 0
                    else:
                        right_count = 0
                        print("migijanai")
                else:
                    right_flg = False
                    right_count = 0

                pre_cx = cx
                pre_cy = cy
        else:
            right_flg = False

        # キー処理(ESC：終了) #################################################
        key = cv.waitKey(1)
        if key == 27:  # ESC
            break

        # 画面反映 #############################################################
        cv.imshow('MediaPipe Hand Demo', debug_image)

    cap.release()
    cv.destroyAllWindows()


def calc_palm_moment(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    palm_array = np.empty((0, 2), int)

    for index, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        if index == 0:  # 手首1
            palm_array = np.append(palm_array, landmark_point, axis=0)
        if index == 1:  # 手首2
            palm_array = np.append(palm_array, landmark_point, axis=0)
        if index == 5:  # 人差指：付け根
            palm_array = np.append(palm_array, landmark_point, axis=0)
        if index == 9:  # 中指：付け根
            palm_array = np.append(palm_array, landmark_point, axis=0)
        if index == 13:  # 薬指：付け根
            palm_array = np.append(palm_array, landmark_point, axis=0)
        if index == 17:  # 小指：付け根
            palm_array = np.append(palm_array, landmark_point, axis=0)
    M = cv.moments(palm_array)
    cx, cy = 0, 0
    if M['m00'] != 0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

    return cx, cy

if __name__ == '__main__':
    main()
