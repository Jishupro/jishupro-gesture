#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import argparse

import cv2 as cv
import numpy as np
import mediapipe as mp

import time
import socketio
from datetime import datetime

import win32gui
import win32con

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
    # サーバに接続 #############################################################
    sio_client = socketio.Client()
    sio_client.connect('http://localhost:5000')

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
    cap = cv.VideoCapture(cap_device, cv.CAP_DSHOW)
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
    cap = cv.VideoCapture(0)
    print(cap.get(cv.CAP_PROP_FPS))
    print(cap.set(cv.CAP_PROP_FPS, 100))
    print(cap.get(cv.CAP_PROP_FPS))
    right_count = 0
    left_count = 0
    right_flg = False
    left_flg = False
    horizontal_threshold = 0.5
    gesture_threshold = 4 # 
    pre_cx = 500
    pre_cy = 500
    init_flg = False
    window_w = 640 - 17
    window_h = 480
    hover_frames = 10
    rect_left_point = [0, int(window_h/2-50), 100, int(window_h/2+50)]
    rect_right_point = [window_w-100, int(window_h/2-50), window_w, int(window_h/2+50)]
    left_rect_center_point = [int((rect_left_point[0]+rect_left_point[2])/2), int((rect_left_point[1]+rect_left_point[3])/2)]
    right_rect_center_point = [int((rect_right_point[0]+rect_right_point[2])/2), int((rect_right_point[1]+rect_right_point[3])/2)]
    while True:
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
        # 手を置くポイントの矩形
        cv.rectangle(debug_image, (rect_left_point[0], rect_left_point[1]), (rect_left_point[2], rect_left_point[3]), (0, 255, 0), 2)
        cv.rectangle(debug_image, (rect_right_point[0], rect_right_point[1]), (rect_right_point[2], rect_right_point[3]), (0, 255, 0), 2)
        if results.multi_hand_landmarks is not None: # 手が認識されたとき
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                  results.multi_handedness):
                
                # 手の平重心計算
                cx, cy = calc_palm_moment(debug_image, hand_landmarks)
                # 外接矩形の計算
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                # 描画
                debug_image = draw_landmarks(debug_image, cx, cy,
                                             hand_landmarks, handedness)
                debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                # 左の矩形に触れた場合
                if rect_left_point[0] <= cx and cx <= rect_left_point[2] and rect_left_point[1] <= cy and cy <= rect_left_point[3]:
                    cv.rectangle(debug_image, (rect_left_point[0], rect_left_point[1]), (rect_left_point[2], rect_left_point[3]), (0, 0, 255), 2)
                    diff = int((rect_left_point[2]-rect_left_point[0])/2*min(left_count, hover_frames)/hover_frames)
                    cv.rectangle(debug_image, (left_rect_center_point[0]-diff, left_rect_center_point[1]-diff),  (left_rect_center_point[0]+diff, left_rect_center_point[1]+diff), (0, 0, 255), -1)
                    left_count += 1
  
                    if hover_frames < left_count and not left_flg:
                        sio_client.emit('moved', {"action":"prev"}) 
                        left_flg = True
                        print("<<<<<<<<<<<<<<<<<<<<<<")
                # 右の矩形に触れた場合
                elif rect_right_point[0] <= cx and cx <= rect_right_point[2] and rect_right_point[1] <= cy and cy <= rect_right_point[3]:
                    cv.rectangle(debug_image, (rect_right_point[0], rect_right_point[1]), (rect_right_point[2], rect_right_point[3]), (0, 0, 255), 2)
                    diff = int((rect_left_point[3]-rect_left_point[1])/2*min(right_count, hover_frames)/hover_frames)
                    cv.rectangle(debug_image, (right_rect_center_point[0]-diff, right_rect_center_point[1]-diff),  (right_rect_center_point[0]+diff, right_rect_center_point[1]+diff), (0, 0, 255), -1)
                    right_count += 1 
                    if hover_frames < right_count and not right_flg:
                        sio_client.emit('moved', {"action":"next"}) 
                        right_flg = True
                        print("                  >>>>>>>>>>>>>>>>>>>>>>>")
                else:
                    cv.rectangle(debug_image, (rect_left_point[0], rect_left_point[1]), (rect_left_point[2], rect_left_point[3]), (0, 255, 0), 2)
                    left_flg = False
                    left_count = 0
                    cv.rectangle(debug_image, (rect_right_point[0], rect_right_point[1]), (rect_right_point[2], rect_right_point[3]), (0, 255, 0), 2)
                    right_flg = False
                    right_count = 0
                
                
                    

                # print("***cx: ", cx, "***cy: ", cy)
                # if cx != pre_cx and abs(cy - pre_cy) / abs(cx - pre_cx) < horizontal_threshold: # 水平方向に動いたとき
                #     if 0 < cx - pre_cx: # 右
                #         right_count += 1 
                #         if gesture_threshold < right_count and not right_flg:
                #             sio_client.emit('moved', {"action":"prev"}) 
                #             right_flg = True
                #             print("                  >>>>>>>>>>>>>>>>>>>>>>>")
                #             right_count = 0
                #     else: # 左
                #         left_count += 1 
                #         if gesture_threshold < left_count and not left_flg:
                #             sio_client.emit('moved', {"action":"next"}) 
                #             left_flg = True
                #             print("<<<<<<<<<<<<<<<<<<<<<<")
                #             left_count = 0
                # else: # 水平方向に動いていないとき
                #     right_flg = False
                #     right_count = 0
                #     left_flg = False
                #     left_count = 0

                # pre_cx = cx
                # pre_cy = cy
        else: # 手が認識されていないとき
            right_flg = False
            left_flg = False

        # 画面反映 #############################################################
        # debug_image = cv.resize(debug_image, dsize=(int(debug_image.shape[1]/2), int(debug_image.shape[0]/2)))
        cv.imshow('MediaPipe Hand Demo', debug_image)

        # キー処理(ESC：終了) #################################################
        key = cv.waitKey(1)
        is_visible = cv.getWindowProperty('MediaPipe Hand Demo', cv.WND_PROP_VISIBLE)
        if key == 27 or not is_visible:  # ESC
            break

        if not init_flg:
            hwnd = win32gui.GetActiveWindow()
            init_flg = True

        (x, y, w, h) = cv.getWindowImageRect('MediaPipe Hand Demo')
        # print(win32gui.GetActiveWindow())
        # win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0, 0, w, h, win32con.SWP_NOACTIVATE)
        win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0, 2045, w, h, win32con.SWP_NOACTIVATE)

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



def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


def draw_landmarks(image, cx, cy, landmarks, handedness):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # キーポイント
    for index, landmark in enumerate(landmarks.landmark):
        if landmark.visibility < 0 or landmark.presence < 0:
            continue

        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append((landmark_x, landmark_y))

        if index == 0:  # 手首1
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 1:  # 手首2
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 2:  # 親指：付け根
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 3:  # 親指：第1関節
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 4:  # 親指：指先
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            cv.circle(image, (landmark_x, landmark_y), 12, (0, 255, 0), 2)
        if index == 5:  # 人差指：付け根
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 6:  # 人差指：第2関節
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 7:  # 人差指：第1関節
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 8:  # 人差指：指先
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            cv.circle(image, (landmark_x, landmark_y), 12, (0, 255, 0), 2)
        if index == 9:  # 中指：付け根
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 10:  # 中指：第2関節
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 11:  # 中指：第1関節
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 12:  # 中指：指先
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            cv.circle(image, (landmark_x, landmark_y), 12, (0, 255, 0), 2)
        if index == 13:  # 薬指：付け根
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 14:  # 薬指：第2関節
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 15:  # 薬指：第1関節
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 16:  # 薬指：指先
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            cv.circle(image, (landmark_x, landmark_y), 12, (0, 255, 0), 2)
        if index == 17:  # 小指：付け根
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 18:  # 小指：第2関節
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 19:  # 小指：第1関節
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 20:  # 小指：指先
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            cv.circle(image, (landmark_x, landmark_y), 12, (0, 255, 0), 2)

    # 接続線
    if len(landmark_point) > 0:
        # 親指
        cv.line(image, landmark_point[2], landmark_point[3], (0, 255, 0), 2)
        cv.line(image, landmark_point[3], landmark_point[4], (0, 255, 0), 2)

        # 人差指
        cv.line(image, landmark_point[5], landmark_point[6], (0, 255, 0), 2)
        cv.line(image, landmark_point[6], landmark_point[7], (0, 255, 0), 2)
        cv.line(image, landmark_point[7], landmark_point[8], (0, 255, 0), 2)

        # 中指
        cv.line(image, landmark_point[9], landmark_point[10], (0, 255, 0), 2)
        cv.line(image, landmark_point[10], landmark_point[11], (0, 255, 0), 2)
        cv.line(image, landmark_point[11], landmark_point[12], (0, 255, 0), 2)

        # 薬指
        cv.line(image, landmark_point[13], landmark_point[14], (0, 255, 0), 2)
        cv.line(image, landmark_point[14], landmark_point[15], (0, 255, 0), 2)
        cv.line(image, landmark_point[15], landmark_point[16], (0, 255, 0), 2)

        # 小指
        cv.line(image, landmark_point[17], landmark_point[18], (0, 255, 0), 2)
        cv.line(image, landmark_point[18], landmark_point[19], (0, 255, 0), 2)
        cv.line(image, landmark_point[19], landmark_point[20], (0, 255, 0), 2)

        # 手の平
        cv.line(image, landmark_point[0], landmark_point[1], (0, 255, 0), 2)
        cv.line(image, landmark_point[1], landmark_point[2], (0, 255, 0), 2)
        cv.line(image, landmark_point[2], landmark_point[5], (0, 255, 0), 2)
        cv.line(image, landmark_point[5], landmark_point[9], (0, 255, 0), 2)
        cv.line(image, landmark_point[9], landmark_point[13], (0, 255, 0), 2)
        cv.line(image, landmark_point[13], landmark_point[17], (0, 255, 0), 2)
        cv.line(image, landmark_point[17], landmark_point[0], (0, 255, 0), 2)

    # 重心 + 左右
    if len(landmark_point) > 0:
        # handedness.classification[0].index
        # handedness.classification[0].score

        cv.circle(image, (cx, cy), 12, (0, 255, 0), 2)
        cv.putText(image, handedness.classification[0].label[0],
                   (cx - 6, cy + 6), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0),
                   2, cv.LINE_AA)  # label[0]:一文字目だけ

    return image


def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        # 外接矩形
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (0, 255, 0), 2)

    return image



if __name__ == '__main__':
    main()
