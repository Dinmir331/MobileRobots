#!/usr/bin/env python3
"""
Ручное управление Robotino мышью с видом сверху.
Поддерживает два источника видео:
  - веб-камера (числовой индекс, например 0)
  - видеофайл (строка пути к файлу)
- Настройка рамки поля (выделение ROI + реальные размеры)
- Визуализация текущих координат робота (из одометрии)
- Отображение пройденного маршрута
- Управление кликом мыши: робот едет в указанную точку
- Дополнительно: на роботе может быть закреплён маркер, для рисования траектории
  на реальном поле (управление подъёмом/опусканием маркера не реализовано,
  но может быть добавлено через API Robotino).
"""

import cv2
import numpy as np
import requests
import time
import threading
import sys
from collections import deque

# ========== НАСТРОЙКИ ==========
ROBOTINO_IP = "192.168.0.1"          # IP робота
# ИСТОЧНИК ВИДЕО: число -> индекс камеры, строка -> путь к файлу
VIDEO_SOURCE = 0                     # или "video.mp4"
MAX_SPEED = 0.2                      # максимальная линейная скорость, м/с
KP = 0.8                             # коэффициент пропорционального регулятора
DIST_THRESHOLD = 0.03                # порог достижения цели, м
ODOMETRY_RATE = 10                   # частота опроса одометрии, Гц
# ================================

# Глобальные переменные для одометрии
robot_lock = threading.Lock()
robot_x, robot_y, robot_phi = 0.0, 0.0, 0.0
robot_odom_valid = False

# Целевая точка (в мировых координатах) и флаг активности
target_world = None
target_lock = threading.Lock()

# История траектории (мировые координаты)
trajectory = deque(maxlen=1000)

# Параметры калибровки ROI
roi = None          # (x, y, w, h) в пикселях
real_width = None   # метры
real_height = None  # метры

def get_odometry():
    """Получить одометрию от Robotino, возвращает [x, y, phi, vx, vy, omega, seq] или None."""
    try:
        url = f"http://{ROBOTINO_IP}/data/odometry"
        resp = requests.get(url, timeout=0.2)
        if resp.status_code == 200:
            data = resp.json()
            if len(data) == 7:
                return data
    except Exception:
        pass
    return None

def send_velocity(vx, vy, omega):
    """Отправить команду скоростей роботу."""
    try:
        url = f"http://{ROBOTINO_IP}/data/omnidrive"
        requests.post(url, json=[vx, vy, omega], timeout=0.2)
    except Exception:
        pass

def odometry_thread():
    """Фоновый поток для непрерывного получения одометрии."""
    global robot_x, robot_y, robot_phi, robot_odom_valid
    while True:
        data = get_odometry()
        if data is not None:
            with robot_lock:
                robot_x, robot_y, robot_phi = data[0], data[1], data[2]
                robot_odom_valid = True
        time.sleep(1.0 / ODOMETRY_RATE)

def pixel_to_world(px, py):
    """Преобразование координат пикселя в мировые (метры) на основе ROI."""
    if roi is None or real_width is None or real_height is None:
        return None
    rx, ry, rw, rh = roi
    # Проверка попадания в ROI
    if not (rx <= px <= rx + rw and ry <= py <= ry + rh):
        return None
    # world X: вправо, world Y: вверх (от нижней границы ROI)
    wx = (px - rx) * (real_width / rw)
    wy = (ry + rh - py) * (real_height / rh)
    return (wx, wy)

def world_to_pixel(wx, wy):
    """Преобразование мировых координат в пиксельные (для рисования)."""
    if roi is None or real_width is None or real_height is None:
        return None
    rx, ry, rw, rh = roi
    px = rx + (wx / real_width) * rw
    py = ry + rh - (wy / real_height) * rh
    return (int(px), int(py))

def draw_interface(frame):
    """Отрисовка ROI, робота, цели и траектории."""
    # ROI
    if roi is not None:
        rx, ry, rw, rh = roi
        cv2.rectangle(frame, (rx, ry), (rx+rw, ry+rh), (255, 255, 0), 2)
        cv2.putText(frame, "ROI", (rx, ry-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)

    # Траектория
    if trajectory:
        pts = []
        for wx, wy in trajectory:
            pt = world_to_pixel(wx, wy)
            if pt:
                pts.append(pt)
        if len(pts) > 1:
            cv2.polylines(frame, [np.array(pts)], False, (0, 255, 0), 2)

    # Текущее положение робота
    with robot_lock:
        cur_x, cur_y, cur_phi = robot_x, robot_y, robot_phi
        valid = robot_odom_valid
    if valid and roi is not None:
        pt = world_to_pixel(cur_x, cur_y)
        if pt:
            cv2.circle(frame, pt, 8, (0, 0, 255), -1)
            arrow_len = 15
            end_pt = (int(pt[0] + arrow_len * np.cos(cur_phi)),
                      int(pt[1] - arrow_len * np.sin(cur_phi)))
            cv2.arrowedLine(frame, pt, end_pt, (0, 0, 255), 2)

    # Целевая точка
    with target_lock:
        tw = target_world
    if tw is not None and roi is not None:
        pt = world_to_pixel(tw[0], tw[1])
        if pt:
            cv2.drawMarker(frame, pt, (255, 0, 0), cv2.MARKER_CROSS, 15, 2)

    return frame

def main():
    global roi, real_width, real_height, target_world

    # Инициализация источника видео
    source = VIDEO_SOURCE
    # Если передан аргумент командной строки, используем его
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        # Попробуем интерпретировать как число (индекс камеры)
        try:
            source = int(arg)
        except ValueError:
            source = arg  # иначе считаем путём к файлу

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Не удалось открыть источник видео: {source}")
        return

    # Запуск потока одометрии
    odom_thread = threading.Thread(target=odometry_thread, daemon=True)
    odom_thread.start()

    cv2.namedWindow("Robotino Control")
    cv2.setMouseCallback("Robotino Control", mouse_callback)

    print("Инструкция:")
    print(" - 'r' : задать ROI (выделите прямоугольник мышью и нажмите Enter/Space)")
    print(" - 'c' : очистить траекторию")
    print(" - 'q' : выход")
    print(" - ЛКМ по изображению : задать целевую точку")
    print("Для калибровки нажмите 'r' и выделите поле.")
    print(f"Источник видео: {'камера' if isinstance(source, int) else source}")

    while True:
        ret, frame = cap.read()
        if not ret:
            # Если видеофайл закончился, перезапустим его
            if isinstance(source, str):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                print("Ошибка захвата кадра с камеры.")
                break

        # Обработка движения к цели
        with target_lock:
            tw = target_world
        if tw is not None:
            with robot_lock:
                rx, ry = robot_x, robot_y
                valid = robot_odom_valid
            if valid:
                dx = tw[0] - rx
                dy = tw[1] - ry
                dist = np.hypot(dx, dy)
                if dist < DIST_THRESHOLD:
                    with target_lock:
                        target_world = None
                    send_velocity(0, 0, 0)
                else:
                    vx = np.clip(KP * dx, -MAX_SPEED, MAX_SPEED)
                    vy = np.clip(KP * dy, -MAX_SPEED, MAX_SPEED)
                    send_velocity(vx, vy, 0)
            else:
                send_velocity(0, 0, 0)
        else:
            send_velocity(0, 0, 0)

        # Сохранение траектории
        with robot_lock:
            if robot_odom_valid:
                trajectory.append((robot_x, robot_y))

        # Визуализация
        frame = draw_interface(frame)
        cv2.imshow("Robotino Control", frame)

        key = cv2.waitKey(20) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            roi_rect = cv2.selectROI("Robotino Control", frame, showCrosshair=True, fromCenter=False)
            if roi_rect[2] > 0 and roi_rect[3] > 0:
                roi = roi_rect
                try:
                    w_str = input("Введите реальную ширину поля (метры): ")
                    h_str = input("Введите реальную высоту поля (метры): ")
                    real_width = float(w_str)
                    real_height = float(h_str)
                    print(f"ROI задан: {roi}, размеры {real_width}x{real_height} м")
                    trajectory.clear()
                    with target_lock:
                        target_world = None
                except ValueError:
                    print("Ошибка ввода чисел. Калибровка не выполнена.")
                    roi = None
            cv2.destroyWindow("ROI selector")
        elif key == ord('c'):
            trajectory.clear()
            print("Траектория очищена.")

    send_velocity(0, 0, 0)
    cap.release()
    cv2.destroyAllWindows()

def mouse_callback(event, x, y, flags, param):
    """Обработчик кликов мыши: установка целевой точки."""
    global target_world
    if event == cv2.EVENT_LBUTTONDOWN:
        if roi is not None:
            wpt = pixel_to_world(x, y)
            if wpt is not None:
                with target_lock:
                    target_world = wpt
                print(f"Цель: X={wpt[0]:.3f} м, Y={wpt[1]:.3f} м")

if __name__ == "__main__":
    main()