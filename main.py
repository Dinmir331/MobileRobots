# ================================================================
#  РАЗДЕЛ 1: ИМПОРТЫ И НАСТРОЙКИ
# ================================================================
# Библиотеки и функции
import cv2
import numpy as np
import time
import math
import heapq
import threading

from collections import deque
from Robotino_communication import send_velocity, get_odometry
import json
import os

import matplotlib.pyplot as plt
import datetime

# Режимы работы
USE_CAMERA = True               # True - камера, False - видеофайл video.mp4
CONTROL = True                  # True - управление роботом, False - только видео-поток
OBSTACLE_MODE = "static"        # "static" - препятствия по первому кадру после калибровки, "dynamic" - обновлять постоянно
PATH_ALGORITHM = "A*"            # Алгоритм построения пути: "A*", "Dijkstra", "Greedy", "Bidirectional"

# SPLINE
PATH_SMOOTHING = True           # True - сглаживать путь сплайнами, False - без сглаживания
SPLINE_RESOLUTION = 0.05        # расстояние между точками сглаженного пути в метрах (чем меньше, тем плавнее)
SIMPLIFY_EPSILON_PX = 15       # допустимое отклонение в пикселях при упрощении (0 — без упрощения)

LOG_DIR = "trajectory_logs"    # папка для сохранения графиков
goal_just_reached = False        # флаг, что цель только что задана и ждёт сохранения

# Видео-поток и передача информации
CAMERA_INDEX = 0                # индекс камеры или путь к видеофайлу
VIDEO_NAME = "video_4.mp4"      # название файла видео
LOOP_VIDEO = False              # зацикливать видеофайл при окончании
SEND_INTERVAL = 0.05             # секунды, минимальный интервал отправки команд
PIX_PER_METER = 200             # Масштаб: сколько пикселей в одном метре на выходном изображении
OBSTACLE_PROCESS_SCALE = 1      # во сколько раз уменьшать изображение для обработки препятствий (ускоряет расчёт)
remap_maps = None               # Карты для cv2.remap

trajectory_world = deque(maxlen=1000)  # очередь пройденных позиций
trajectory_time = deque(maxlen=1000)   # метки времени для каждой позиции

# распознавание цветов на картинке // ЦВЕТОВЫЕ ДИАПАЗОНЫ ПРЕПЯТСТВИЙ (HSV)
# Каждый диапазон: (H_low, S_low, V_low, H_high, S_high, V_high)
# H от 0 до 179, S и V от 0 до 255
#OBSTACLE_COLORS = [
    # Чёрный: только ограничение по яркости V_high = 100
#    (0, 0, 0, 179, 255, 180),
    # Красный 1 диапазон (H: 0-10, S: 60-255, V: 50-255)
 #   (0, 60, 50, 10, 255, 255),
    # Красный 2 диапазон (H: 100-179, S: 60-255, V: 40-255)
    # (H_high=180 скорректирован до 179 – это максимум в OpenCV)
#    (100, 60, 40, 179, 255, 255),
    # Зелёный (H: 30-100, S: 60-255, V: 30-255)
 #   (30, 60, 30, 100, 255, 255),
    # Синий (H: 80-140, S: 70-255, V: 50-255)
#    (80, 70, 50, 140, 255, 255)
#]

OBSTACLE_COLORS = [
     (0, 0, 0, 0, 0, 0), # Чёрный (низкая яркость) V > 30 - серый
     (0, 50, 50, 10, 255, 255), # Красный (1 диапазон)
     (160, 50, 50, 180, 255, 255), # Красный (2 диапазон)
     (40, 50, 50, 80, 255, 255), # Зелёный
     (100, 50, 50, 130, 255, 255) # Синий
]
# Морфология для очистки маски
MORPH_OPEN_KERNEL_SIZE = 5        # размер ядра открытия (убрать шум)
MORPH_CLOSE_KERNEL_SIZE = 15      # размер ядра закрытия (заполнить дыры)

# распознавание препятствий
MIN_OBSTACLE_AREA_PX = 150      # минимальная площадь препятствия в пикселях (меньше — игнорируем)
OBSTACLE_MERGE_DIST_PX = 5      # расстояние в пикселях для слияния близких препятствий
PROCESS_EVERY_N_FRAMES = 30      # обрабатывать препятствия каждый N-й кадр (1 — каждый)
BORDER_MARGIN_M = 0.15           # отступ от края поля в метрах
OBSTACLE_SAFE_RADIUS_M = 0.3    # безопасный радиус вокруг препятствий в метрах (не меняется при масштабе)
PLANNING_EXTRA_RADIUS_M = 0.001   # дополнительный отступ при построении маршрута (метры)

# Габариты робота и поля
ROBOT_DIAMETER_M = 0.6      # диаметр робота в метрах (для исключения из препятствий)
FIELD_WIDTH = 2.2               # метры, реальная ширина поля
FIELD_HEIGHT = 2.2              # метры, реальная высота поля

# Движение и скорость робота
MAX_SPEED = 0.2                 # м/с, максимальная линейная скорость
GOAL_TOLERANCE = 0.1           # метры, радиус достижения целевой траектории
WP_THRESHOLD_M = 0.1           # м до waypoint'а — переключаемся на следующий

# ================================================================
#  РАЗДЕЛ 2: ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ СОСТОЯНИЯ
# ================================================================

# Точки границ поля
calib_points = []               # список 4 углов поля в пикселях
M = None                        # матрица гомографии (пиксели -> мировые метры)

# Траектори
target_world = None             # целевая точка в мировых координатах (x, y)
trajectory_world = deque(maxlen=1000)  # очередь пройденных позиций
robot_pos = None                # (x_world, y_world, angle_rad)
planned_path = []               # список пиксельных точек пути (в полном разрешении)
goal_prev = None                # предыдущая цель для отслеживания смены
waypoint_index = 0              # текущий индекс waypoint в planned_path
last_wp_index = 0               # индекс последней выбранной waypoint'ы
last_send_time = 0              # Время

# Картинка
warp_window_initialized = False
frame_count = 0
last_binary_obs = None          # хранит последнюю рассчитанную маску препятствий
static_binary_mask = None       # маска для статического режима
plot_saved = False              # флаг, что график для текущей цели уже сохранён
initial_planned_path = []        # сохраняет первый план для текущей цели

# Одометрия
odom_text = ""                  # строка с данными одометрии для отображения
current_vx_world = 0.0          # текущая мировая скорость X (вниз)
current_vy_world = 0.0          # текущая мировая скорость Y (вправо)
last_odom_time = 0              # одометрия

# ================================================================
#  РАЗДЕЛ 3: ИНИЦИАЛИЗАЦИЯ ARUCO ДЕТЕКТОРА
# ================================================================
MARKER_ID = 9                         # ID ArUco маркера
ARUCO_DICT = cv2.aruco.DICT_6X6_250

aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
aruco_params = cv2.aruco.DetectorParameters()
# Настройки для более стабильного распознавания
aruco_params.adaptiveThreshWinSizeMin = 3
aruco_params.adaptiveThreshWinSizeMax = 23
aruco_params.adaptiveThreshWinSizeStep = 10
aruco_params.adaptiveThreshConstant = 7
aruco_params.minMarkerPerimeterRate = 0.03
aruco_params.maxMarkerPerimeterRate = 4.0
aruco_params.polygonalApproxAccuracyRate = 0.03
aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_NONE

# ================================================================
#  РАЗДЕЛ 4: ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ================================================================

##########################
# Сохранение точек калиббровки поля по камере
CALIB_FILE = "field_calibration.json"

def save_calibration():
    """Сохраняет calib_points в JSON-файл."""
    try:
        with open(CALIB_FILE, 'w') as f:
            json.dump(calib_points, f)
        print("Калибровочные точки сохранены.")
    except Exception as e:
        print(f"Ошибка сохранения калибровки: {e}")

def load_calibration():
    """Загружает точки из файла и вычисляет M. Возвращает True при успехе."""
    global M, calib_points
    if not os.path.exists(CALIB_FILE):
        return False
    try:
        with open(CALIB_FILE, 'r') as f:
            pts = json.load(f)
        if len(pts) != 4:
            return False
        calib_points = [tuple(p) for p in pts]  # список кортежей (x,y)
        # Вычисляем гомографию
        ww = int(FIELD_WIDTH * PIX_PER_METER)
        wh = int(FIELD_HEIGHT * PIX_PER_METER)
        dst_corners = np.array([[0, 0], [ww, 0], [ww, wh], [0, wh]], dtype=np.float32)
        src_corners = np.array(calib_points, dtype=np.float32)
        M, _ = cv2.findHomography(src_corners, dst_corners)
        if M is not None:
            print("Калибровка загружена из файла.")
            return True
        else:
            calib_points = []
            return False
    except Exception as e:
        print(f"Ошибка загрузки калибровки: {e}")
        return False

##########################
def safe_send(vx, vy, omega):
    def _send():
        try:
            send_velocity(float(vx), float(vy), float(omega))
        except Exception:
            pass
    
    # Запускаем отправку в фоновом потоке, чтобы не блокировать главный цикл
    threading.Thread(target=_send, daemon=True).start()

# Обработчик мыши
def mouse_callback(event, x, y, flags, param):
    global calib_points, target_world, M
    if event == cv2.EVENT_LBUTTONDOWN:
        if M is None and len(calib_points) < 4:
            calib_points.append((x, y))
            if len(calib_points) == 4:
                ww = int(FIELD_WIDTH * PIX_PER_METER)
                wh = int(FIELD_HEIGHT * PIX_PER_METER)
                dst_corners = np.array([[0, 0], [ww, 0], [ww, wh], [0, wh]], dtype=np.float32)
                src_corners = np.array(calib_points, dtype=np.float32)
                M, _ = cv2.findHomography(src_corners, dst_corners)
                if M is None:
                    calib_points = []
                    static_binary_mask = None
                else:
                    global remap_maps
                    # Предварительно считаем карты для remap
                    h_out = int(FIELD_HEIGHT * PIX_PER_METER)
                    w_out = int(FIELD_WIDTH * PIX_PER_METER)
                    remap_maps = cv2.convertMaps(
                        *cv2.initUndistortRectifyMap(
                            np.eye(3), None, None, (w_out, h_out), cv2.CV_32FC1
                        ), cv2.CV_16SC2
                    )
                    # Для перспективного преобразования лучше использовать стандартный метод генерации карт:
                    remap_maps = cv2.initUndistortRectifyMap(...) # Нет, для гомографии проще:
                    
                    # Правильный способ для гомографии:
                    grid_x, grid_y = np.meshgrid(np.arange(w_out), np.arange(h_out))
                    # Инверсная гомография (из warped в original)
                    M_inv = np.linalg.inv(M)
                    pts = np.float32([[grid_x.flatten(), grid_y.flatten(), np.ones_like(grid_x).flatten()]]).transpose().reshape(-1, 1, 3)
                    src_pts = cv2.perspectiveTransform(pts, M_inv)
                    map_x = src_pts[:,:,0].reshape(h_out, w_out).astype(np.float32)
                    map_y = src_pts[:,:,1].reshape(h_out, w_out).astype(np.float32)
                    remap_maps = (map_x, map_y)
                    
                    save_calibration()
        elif M is not None:
            # Клик по warped-изображению: X пикселя -> мировая Y (вправо), Y пикселя -> мировая X (вниз)
            target_world = (y / PIX_PER_METER, x / PIX_PER_METER)
            goal_just_reached = True          # новая цель → при достижении сохраним
            trajectory_world.clear()          # очищаем старую траекторию
            trajectory_time.clear()
            initial_planned_path = []
            plot_saved = False

# Детекция робота по ArUco маркеру
def detect_robot(frame):
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    corners, ids, _ = detector.detectMarkers(frame)
    if ids is not None and MARKER_ID in ids.flatten():
        idx = np.where(ids.flatten() == MARKER_ID)[0][0]
        corner = corners[idx][0]
        center = np.mean(corner, axis=0)
        vec = corner[1] - corner[0]
        angle = math.atan2(vec[1], vec[0])
        # Преобразуем угол в мировую систему: X вниз, Y вправо
        phi = - angle
        return center[0], center[1], phi, corner   # возвращаем phi
    return None

# ================================================================
#  РАЗДЕЛ 5: ФУНКЦИИ ОТРИСОВКИ
# ================================================================

def draw_calibration_points(frame):
    for pt in calib_points:
        cv2.circle(frame, pt, 5, (0, 0, 0), -1)   # чёрный цвет

def draw_field(frame):
    if len(calib_points) == 4:
        pts = np.array(calib_points, np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=1)

def draw_trajectory(frame):
    if M is None or len(trajectory_world) < 2:
        return
    # Мировые координаты (x, y): x вниз, y вправо
    # Пиксель X = мировая Y * PIX_PER_METER, Пиксель Y = мировая X * PIX_PER_METER
    pts = [(int(p[1] * PIX_PER_METER), int(p[0] * PIX_PER_METER)) for p in trajectory_world]
    for i in range(1, len(pts)):
        cv2.line(frame, pts[i-1], pts[i], (0, 140, 255), 2)   # оранжевый

def draw_status_info(frame, robot_pos, target_world, vx_w, vy_w):
    """Выводит информацию о скоростях и позиции в правом верхнем углу кадра."""
    if robot_pos is None:
        return
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    color = (0, 0, 0)  # чёрный текст

    # Собираем строки
    lines = []
    sp = math.hypot(vx_w, vy_w)
    lines.append(f"Speed: {sp:.2f} m/s")
    lines.append(f"Vx: {vx_w:.2f}  Vy: {vy_w:.2f}")
    if robot_pos is not None:
        rx, ry, _ = robot_pos
        lines.append(f"Pos R: ({rx:.2f}, {ry:.2f})")
    else:
        lines.append("Pos R: ---")
    if target_world is not None:
        gx, gy = target_world
        lines.append(f"Goal: ({gx:.2f}, {gy:.2f})")
    else:
        lines.append("Goal: ---")

    # Вычисляем максимальную ширину текста для выравнивания по правому краю
    max_width = 0
    for line in lines:
        size = cv2.getTextSize(line, font, font_scale, thickness)[0]
        max_width = max(max_width, size[0])
    x_start = w - max_width - 10   # отступ 10 пикселей от правого края

    y = 30
    for line in lines:
        cv2.putText(frame, line, (x_start, y), font, font_scale, color, thickness)
        y += 20
# ================================================================
#  РАЗДЕЛ 6: ФУНКЦИИ ОБРАБОТКИ ПРЕПЯТСТВИЙ
# ================================================================

def simplify_obstacle_mask(binary_mask, merge_dist_px):
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros_like(binary_mask)

    # Фильтруем по минимальной площади + выпуклая оболочка
    hulls = [cv2.convexHull(cnt) for cnt in contours if cv2.contourArea(cnt) >= MIN_OBSTACLE_AREA_PX]
    if not hulls:
        return np.zeros_like(binary_mask)

    temp = np.zeros_like(binary_mask)
    cv2.drawContours(temp, hulls, -1, 255, cv2.FILLED)

    # Слияние близких оболочек
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (merge_dist_px * 2 + 1, merge_dist_px * 2 + 1))
    merged = cv2.dilate(temp, kernel, iterations=1)

    merged_contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    simplified = np.zeros_like(binary_mask)
    # Финальные оболочки тоже фильтруем по площади
    final_hulls = [cv2.convexHull(cnt) for cnt in merged_contours if cv2.contourArea(cnt) >= MIN_OBSTACLE_AREA_PX]
    cv2.drawContours(simplified, final_hulls, -1, 255, cv2.FILLED)
    return simplified

def detect_obstacles(warped_frame, robot_pix=None, marker_corners=None, scale=1):
    """
    Создаёт бинарную маску препятствий на основе заданных цветовых диапазонов.
    Возвращает маску того же размера, что и warped_frame.
    """
    h_full, w_full = warped_frame.shape[:2]
    # Работаем на уменьшенном изображении для скорости
    if scale > 1:
        small_h, small_w = int(h_full / scale), int(w_full / scale)
        clean = cv2.resize(warped_frame, (small_w, small_h), interpolation=cv2.INTER_AREA)
        if robot_pix is not None:
            rx_small = int(robot_pix[0] / scale)
            ry_small = int(robot_pix[1] / scale)
            robot_pix_small = (rx_small, ry_small)
        else:
            robot_pix_small = None
        if marker_corners is not None:
            marker_corners_small = [(int(x/scale), int(y/scale)) for (x, y) in marker_corners]
        else:
            marker_corners_small = None
    else:
        clean = warped_frame.copy()
        robot_pix_small = robot_pix
        marker_corners_small = marker_corners

    # Исключаем маркер и корпус робота
    if marker_corners_small is not None:
        pts = np.array(marker_corners_small, dtype=np.int32)
        cv2.fillPoly(clean, [pts], (255, 255, 255))
    if robot_pix_small is not None:
        rx, ry = robot_pix_small
        radius_px = int((ROBOT_DIAMETER_M / 2) * PIX_PER_METER / scale)
        cv2.circle(clean, (rx, ry), radius_px, (255, 255, 255), -1)

    # Переводим в HSV для цветовой фильтрации
    hsv = cv2.cvtColor(clean, cv2.COLOR_BGR2HSV)

    # Собираем маски по всем заданным диапазонам
    combined_mask = np.zeros((clean.shape[0], clean.shape[1]), dtype=np.uint8)
    for color_range in OBSTACLE_COLORS:
        # color_range содержит 6 чисел: (H_low, S_low, V_low, H_high, S_high, V_high)
        lower = np.array(color_range[:3])
        upper = np.array(color_range[3:])
        mask = cv2.inRange(hsv, lower, upper)
        combined_mask = cv2.bitwise_or(combined_mask, mask)

    # Морфологическая очистка
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                           (MORPH_OPEN_KERNEL_SIZE, MORPH_OPEN_KERNEL_SIZE))
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel_open)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                            (MORPH_CLOSE_KERNEL_SIZE, MORPH_CLOSE_KERNEL_SIZE))
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel_close)

    # Упрощение до выпуклых оболочек
    simplified = simplify_obstacle_mask(combined_mask,
                                        max(1, int(OBSTACLE_MERGE_DIST_PX // scale)))

    # Фильтр по минимальной площади
    if scale > 1:
        min_area_scaled = max(1, MIN_OBSTACLE_AREA_PX // (scale * scale))
    else:
        min_area_scaled = MIN_OBSTACLE_AREA_PX
    contours, _ = cv2.findContours(simplified, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered = np.zeros_like(simplified)
    for cnt in contours:
        if cv2.contourArea(cnt) >= min_area_scaled:
            cv2.drawContours(filtered, [cnt], -1, 255, cv2.FILLED)

    # Возвращаем к полному размеру
    if scale > 1:
        filtered = cv2.resize(filtered, (w_full, h_full), interpolation=cv2.INTER_NEAREST)
    return filtered

def draw_obstacles(frame, binary_mask, expanded_mask=None):
    if binary_mask is None:
        return

    # 1. Рисуем сами препятствия (красные) через наложение маски
    # Создаем пустой слой того же размера
    overlay_obs = np.zeros_like(frame)
    overlay_obs[binary_mask > 0] = (0, 0, 255) # Красный цвет (BGR)
    cv2.addWeighted(frame, 1.0, overlay_obs, 0.5, 0, frame) # Полупрозрачное наложение

    # 2. Рисуем безопасную зону (оранжевые), если маска есть
    if expanded_mask is None:
        expand_px = int(OBSTACLE_SAFE_RADIUS_M * PIX_PER_METER)
        kernel_size = int(expand_px * 2 + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        expanded_mask = cv2.dilate(binary_mask, kernel, iterations=1)

    overlay_safe = np.zeros_like(frame)
    # Рисуем только ту часть safe_zone, которая НЕ является самим препятствием
    safe_zone_only = (expanded_mask > 0) & (binary_mask == 0)
    overlay_safe[safe_zone_only] = (255, 100, 0) # Оранжевый цвет (BGR)
    cv2.addWeighted(frame, 1.0, overlay_safe, 0.4, 0, frame)

def draw_border(frame):
    """Пунктирная граница (тёмно-серая)"""
    h, w = frame.shape[:2]
    m = int(BORDER_MARGIN_M * PIX_PER_METER)
    pts = [(m, m), (w-m, m), (w-m, h-m), (m, h-m)]
    for i in range(4):
        cv2.line(frame, pts[i], pts[(i+1)%4], (100, 100, 100), 2, cv2.LINE_AA)

def compute_expanded_mask(binary_mask):
    if binary_mask is None:
        return None
    expand_px = int(OBSTACLE_SAFE_RADIUS_M * PIX_PER_METER)
    kernel_size = int(expand_px * 2 + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    expanded = cv2.dilate(binary_mask, kernel, iterations=1)
    return expanded

# ================================================================
#  РАЗДЕЛ 7: ПЛАНИРОВАНИЕ ПУТИ (A*, Dijkstra, Greedy, D*)
# ================================================================

def save_trajectory_plot(trajectory_world, trajectory_time, goal_world,
                         planned_path, pix_per_meter,
                         algorithm, smoothing):
    """
    Сохраняет линейные графики переходных характеристик X(t) и Y(t)
    с реальными и модельными (запланированными) координатами.
    Модельное время масштабируется под длительность реального движения.
    """
    if not trajectory_world or not trajectory_time:
        return

    os.makedirs(LOG_DIR, exist_ok=True)

    # ----- реальные данные -----
    traj_x = [p[0] for p in trajectory_world]   # мировые X
    traj_y = [p[1] for p in trajectory_world]   # мировые Y
    t = list(trajectory_time)
    t0 = t[0]
    t_rel = [ti - t0 for ti in t]

    # ----- модельная траектория (по planned_path) -----
    t_model, x_model, y_model = [], [], []
    if planned_path and len(planned_path) >= 2:
        # перевод пикселей planned_path в мировые координаты
        world_path = [(p[1] / pix_per_meter, p[0] / pix_per_meter)
                      for p in planned_path]   # (X_world, Y_world)
        # расчёт кумулятивных расстояний
        dist = [0.0]
        for i in range(1, len(world_path)):
            d = math.hypot(world_path[i][0] - world_path[i-1][0],
                           world_path[i][1] - world_path[i-1][1])
            dist.append(dist[-1] + d)
        # исходное модельное время при идеальной скорости MAX_SPEED
        t_model_raw = [d / MAX_SPEED for d in dist]

        # --- МАСШТАБИРОВАНИЕ ПОД РЕАЛЬНОЕ ВРЕМЯ ---
        if t_rel and t_rel[-1] > 0 and t_model_raw[-1] > 0:
            scale_factor = t_rel[-1] / t_model_raw[-1]
        else:
            scale_factor = 1.0
        t_model = [t * scale_factor for t in t_model_raw]
        # -------------------------------------------

        x_model = [p[0] for p in world_path]
        y_model = [p[1] for p in world_path]

    # ----- цель -----
    goal_x = goal_world[0]
    goal_y = goal_world[1]

    # ----- построение -----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # График X(t)
    ax1.plot(t_rel, traj_x, 'b-', linewidth=1.5, label='Real X')
    if t_model:
        ax1.plot(t_model, x_model, 'c--', linewidth=1.5, label='Model X (planned)')
    ax1.axhline(y=goal_x, color='g', linestyle=':', label=f'Goal X={goal_x:.2f}')
    ax1.set_xlabel('Time, s')
    ax1.set_ylabel('X, m')
    ax1.set_title('Step response X(t)')
    ax1.grid(True)
    ax1.legend()

    # График Y(t)
    ax2.plot(t_rel, traj_y, 'r-', linewidth=1.5, label='Real Y')
    if t_model:
        ax2.plot(t_model, y_model, 'm--', linewidth=1.5, label='Model Y (planned)')
    ax2.axhline(y=goal_y, color='g', linestyle=':', label=f'Goal Y={goal_y:.2f}')
    ax2.set_xlabel('Time, s')
    ax2.set_ylabel('Y, m')
    ax2.set_title('Step response Y(t)')
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()

    # ----- сохранение -----
    safe_algo = algorithm.replace('*', 'star')
    algo_str = f"{safe_algo}{'_smooth' if smoothing else ''}"
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(LOG_DIR, f"transient_{algo_str}_{timestamp}.png")
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Transition graph saved: {filename}")
    
def simplify_path_dp(points, epsilon):
    """Упрощает ломаную алгоритмом Дугласа–Пекера. epsilon — макс. расстояние в пикселях."""
    if len(points) < 3 or epsilon <= 0:
        return points
    pts = np.array(points, dtype=np.float32)
    # Используем встроенный approxPolyDP (он работает с целыми точками, поэтому временно округляем)
    pts_int = pts.astype(np.int32).reshape((-1, 1, 2))
    simplified = cv2.approxPolyDP(pts_int, epsilon, closed=False)
    return [tuple(p[0]) for p in simplified]

def smooth_path_catmull_rom(path, resolution_m, pix_per_meter):
    """Возвращает гладкий путь с шагом примерно resolution_m (метры)."""
    if len(path) < 2:
        return path

    # --- 1) Линейная интерполяция до равномерного шага -----------------
    target_step = resolution_m * pix_per_meter
    interpolated = [path[0]]
    for i in range(1, len(path)):
        x1, y1 = path[i-1]
        x2, y2 = path[i]
        seg_len = math.hypot(x2 - x1, y2 - y1)
        if seg_len < target_step:
            interpolated.append(path[i])
            continue
        num = max(1, int(seg_len / target_step) + 1)
        for k in range(1, num):
            t = k / num
            x = x1 + (x2 - x1) * t
            y = y1 + (y2 - y1) * t
            interpolated.append((x, y))
    if len(interpolated) < 2 or (abs(interpolated[-1][0] - path[-1][0]) > 1e-3 or
                                 abs(interpolated[-1][1] - path[-1][1]) > 1e-3):
        interpolated.append(path[-1])

    # --- 2) Сглаживание Catmull‑Rom по интерполированному пути ---------
    MIN_POINTS_PER_SEGMENT = 6
    pts = np.array(interpolated, dtype=np.float32)
    if len(pts) < 2:
        return interpolated

    pts_ext = np.vstack([pts[0], pts, pts[-1]])
    dense = []
    for i in range(len(pts) - 1):
        p0 = pts_ext[i]
        p1 = pts_ext[i+1]
        p2 = pts_ext[i+2]
        p3 = pts_ext[i+3]
        seg_len = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        num = max(MIN_POINTS_PER_SEGMENT, int(seg_len / target_step) + 1)
        for t in np.linspace(0, 1, num, endpoint=False):
            t2 = t * t
            t3 = t2 * t
            x = 0.5 * ((2*p1[0]) + (-p0[0] + p2[0]) * t +
                       (2*p0[0] - 5*p1[0] + 4*p2[0] - p3[0]) * t2 +
                       (-p0[0] + 3*p1[0] - 3*p2[0] + p3[0]) * t3)
            y = 0.5 * ((2*p1[1]) + (-p0[1] + p2[1]) * t +
                       (2*p0[1] - 5*p1[1] + 4*p2[1] - p3[1]) * t2 +
                       (-p0[1] + 3*p1[1] - 3*p2[1] + p3[1]) * t3)
            dense.append((x, y))
    dense.append(pts[-1])

    # --- 3) Финальный ресамплинг для однородности --------------------
    if not dense:
        return dense

    resampled = [dense[0]]
    accum = 0.0
    for i in range(1, len(dense)):
        dx = dense[i][0] - dense[i-1][0]
        dy = dense[i][1] - dense[i-1][1]
        seg_len = math.hypot(dx, dy)
        if seg_len == 0:
            continue
        while accum + seg_len >= target_step:
            ratio = (target_step - accum) / seg_len
            x = dense[i-1][0] + dx * ratio
            y = dense[i-1][1] + dy * ratio
            resampled.append((x, y))
            accum = 0.0
            seg_len -= target_step
            dense[i-1] = (x, y)
        accum += seg_len
    if math.hypot(resampled[-1][0] - dense[-1][0], resampled[-1][1] - dense[-1][1]) > 1e-3:
        resampled.append(dense[-1])

    return resampled

# Вспомогательные функции для поиска пути
def reconstruct_path(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path

def get_neighbors(idx, w, h):
    """Возвращает соседей для одномерного индекса."""
    x, y = idx % w, idx // w
    neighbors = []
    # Соседи: (dx, dy, cost)
    for dx, dy, cost in [(-1,0,1),(1,0,1),(0,-1,1),(0,1,1),(-1,-1,1.414),(-1,1,1.414),(1,-1,1.414),(1,1,1.414)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < w and 0 <= ny < h:
            neighbors.append((ny * w + nx, cost)) # Возвращаем 1D индекс
    return neighbors

def astar_search(small_mask, start, goal):
    h, w = small_mask.shape
    start_idx = start[1] * w + start[0]
    goal_idx = goal[1] * w + goal[0]
    
    open_set = []
    heapq.heappush(open_set, (0, start_idx))
    came_from = {}
    g_score = {start_idx: 0}
    flat_mask = small_mask.flatten()

    while open_set:
        _, current = heapq.heappop(open_set)
        
        if current == goal_idx:
            path_idx = reconstruct_path(came_from, current)
            return [(idx % w, idx // w) for idx in path_idx]

        for neighbor_idx, cost in get_neighbors(current, w, h): # Исправлена опечатка get_neighbors_1d
            if flat_mask[neighbor_idx] != 0:
                continue
                
            tentative_g = g_score[current] + cost
            if neighbor_idx not in g_score or tentative_g < g_score[neighbor_idx]:
                came_from[neighbor_idx] = current
                g_score[neighbor_idx] = tentative_g
                
                nx, ny = neighbor_idx % w, neighbor_idx // w
                f_score = tentative_g + math.hypot(goal[0] - nx, goal[1] - ny)
                heapq.heappush(open_set, (f_score, neighbor_idx))
    return []

def dijkstra_search(small_mask, start, goal):
    h, w = small_mask.shape
    start_idx = start[1] * w + start[0]
    goal_idx = goal[1] * w + goal[0]
    
    open_set = []
    heapq.heappush(open_set, (0, start_idx))
    came_from = {}
    g_score = {start_idx: 0}
    flat_mask = small_mask.flatten()

    while open_set:
        _, current = heapq.heappop(open_set)
        
        if current == goal_idx:
            path_idx = reconstruct_path(came_from, current)
            return [(idx % w, idx // w) for idx in path_idx]

        for neighbor_idx, cost in get_neighbors(current, w, h):
            if flat_mask[neighbor_idx] != 0:
                continue
            tentative_g = g_score[current] + cost
            if neighbor_idx not in g_score or tentative_g < g_score[neighbor_idx]:
                came_from[neighbor_idx] = current
                g_score[neighbor_idx] = tentative_g
                heapq.heappush(open_set, (g_score[neighbor_idx], neighbor_idx))
    return []

def greedy_search(small_mask, start, goal):
    h, w = small_mask.shape
    start_idx = start[1] * w + start[0]
    goal_idx = goal[1] * w + goal[0]
    
    start_h = math.hypot(goal[0]-start[0], goal[1]-start[1])
    open_set = []
    heapq.heappush(open_set, (start_h, start_idx))
    came_from = {}
    closed_set = set()
    flat_mask = small_mask.flatten()

    while open_set:
        _, current = heapq.heappop(open_set)
        
        if current == goal_idx:
            path_idx = reconstruct_path(came_from, current)
            return [(idx % w, idx // w) for idx in path_idx]
            
        if current in closed_set:
            continue
        closed_set.add(current)

        for neighbor_idx, _ in get_neighbors(current, w, h):
            if flat_mask[neighbor_idx] != 0:
                continue
            if neighbor_idx not in came_from and neighbor_idx not in closed_set:
                came_from[neighbor_idx] = current
                nx, ny = neighbor_idx % w, neighbor_idx // w
                priority = math.hypot(goal[0]-nx, goal[1]-ny)
                heapq.heappush(open_set, (priority, neighbor_idx))
    return []

def bidirectional_search(small_mask, start, goal):
    h, w = small_mask.shape
    start_idx = start[1] * w + start[0]
    goal_idx = goal[1] * w + goal[0]

    open_fwd = []
    heapq.heappush(open_fwd, (0, start_idx))
    g_fwd = {start_idx: 0}
    parent_fwd = {}

    open_bwd = []
    heapq.heappush(open_bwd, (0, goal_idx))
    g_bwd = {goal_idx: 0}
    parent_bwd = {}

    intersection = None
    best_cost = float('inf')
    flat_mask = small_mask.flatten()

    while open_fwd or open_bwd:
        if open_fwd:
            _, current = heapq.heappop(open_fwd)

            if current in g_bwd:
                total_cost = g_fwd[current] + g_bwd[current]
                if total_cost < best_cost:
                    best_cost = total_cost
                    intersection = current

            for neighbor_idx, cost in get_neighbors(current, w, h):
                if flat_mask[neighbor_idx] != 0:
                    continue
                tentative_g = g_fwd[current] + cost
                if neighbor_idx not in g_fwd or tentative_g < g_fwd[neighbor_idx]:
                    parent_fwd[neighbor_idx] = current
                    g_fwd[neighbor_idx] = tentative_g
                    nx, ny = neighbor_idx % w, neighbor_idx // w
                    priority = tentative_g + math.hypot(goal[0]-nx, goal[1]-ny)
                    heapq.heappush(open_fwd, (priority, neighbor_idx))

        if open_bwd:
            _, current = heapq.heappop(open_bwd)

            if current in g_fwd:
                total_cost = g_fwd[current] + g_bwd[current]
                if total_cost < best_cost:
                    best_cost = total_cost
                    intersection = current

            for neighbor_idx, cost in get_neighbors(current, w, h):
                if flat_mask[neighbor_idx] != 0:
                    continue
                tentative_g = g_bwd[current] + cost
                if neighbor_idx not in g_bwd or tentative_g < g_bwd[neighbor_idx]:
                    parent_bwd[neighbor_idx] = current
                    g_bwd[neighbor_idx] = tentative_g
                    nx, ny = neighbor_idx % w, neighbor_idx // w
                    priority = tentative_g + math.hypot(start[0]-nx, start[1]-ny)
                    heapq.heappush(open_bwd, (priority, neighbor_idx))

        if intersection is not None:
            if (not open_fwd or open_fwd[0][0] >= best_cost) and \
               (not open_bwd or open_bwd[0][0] >= best_cost):
                break

    if intersection is None:
        return []

    path_fwd = reconstruct_path(parent_fwd, intersection)
    path_bwd = reconstruct_path(parent_bwd, intersection)
    path_idx = path_fwd[:-1] + path_bwd[::-1]
    return [(idx % w, idx // w) for idx in path_idx]

# Основная функция планирования
def plan_path(expanded_mask, start_px, goal_px, plan_scale=4):
    """Планирует путь на уменьшенной маске выбранным алгоритмом."""
    if expanded_mask is None:
        return []

    small_mask = cv2.resize(expanded_mask, None, fx=1/plan_scale, fy=1/plan_scale,
                            interpolation=cv2.INTER_NEAREST)
    h_s, w_s = small_mask.shape
    start = (int(start_px[0] / plan_scale), int(start_px[1] / plan_scale))
    goal = (int(goal_px[0] / plan_scale), int(goal_px[1] / plan_scale))

    if not (0 <= start[0] < w_s and 0 <= start[1] < h_s):
        return []
    if not (0 <= goal[0] < w_s and 0 <= goal[1] < h_s):
        return []

    if small_mask[start[1], start[0]] != 0 or small_mask[goal[1], goal[0]] != 0:
        return []

    # Выбор алгоритма
    if PATH_ALGORITHM == "A*":
        path = astar_search(small_mask, start, goal)
    elif PATH_ALGORITHM == "Dijkstra":
        path = dijkstra_search(small_mask, start, goal)
    elif PATH_ALGORITHM == "Greedy":
        path = greedy_search(small_mask, start, goal)
    elif PATH_ALGORITHM == "Bidirectional":
        path = bidirectional_search(small_mask, start, goal)
    else:
        path = astar_search(small_mask, start, goal)

    if not path:
        return []

    path_full = [(p[0]*plan_scale, p[1]*plan_scale) for p in path]
    return path_full

# ================================================================
#  РАЗДЕЛ 8: ГЛАВНЫЙ ЦИКЛ
# ================================================================

def main():
    # ================================================================
    #  РАЗДЕЛ 8.1: все глобальные переменные
    # ================================================================
    global M, calib_points, target_world, robot_pos, trajectory_world, last_send_time
    global warp_window_initialized, planned_path, goal_prev, waypoint_index, last_odom_time 
    global current_vx_world, current_vy_world, last_wp_index, goal_just_reached, plot_saved
    global initial_planned_path
    
    goal_just_reached = False
    plot_saved = False
            
    frame_count = 0
    last_binary_obs = None
    static_binary_mask = None
    
    planned_path = []
    goal_prev = None
    waypoint_index = 0
    
    current_vx_world = 0.0
    current_vy_world = 0.0
    
    # ================================================================
    #  РАЗДЕЛ 8.2: Видео
    # ================================================================
    # Открытие источника видео
    if USE_CAMERA:
        # Пробуем открыть с DirectShow (стабильнее на Windows)
        cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
        if not cap.isOpened():
            # fallback на авто-бэкенд
            cap = cv2.VideoCapture(CAMERA_INDEX)
    else:
        cap = cv2.VideoCapture(VIDEO_NAME)
    
    if not cap.isOpened():
        print(f"Ошибка: не удалось открыть {'камеру ' + str(CAMERA_INDEX) if USE_CAMERA else VIDEO_NAME}")
        time.sleep(3)
        return
    
    # ------------------- Диалог загрузки калибровки -------------------
    if os.path.exists(CALIB_FILE):
        dialog_frame = np.zeros((200, 800, 3), dtype=np.uint8)
        cv2.putText(dialog_frame, "Press 'y' to recalibrate,", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(dialog_frame, "any other key to load.", (20, 140),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow("Calibration Dialog", dialog_frame)
        key = cv2.waitKey(0) & 0xFF   # ждём нажатия бесконечно
        cv2.destroyWindow("Calibration Dialog")

        # 'y' (латинская) или 'н' (русская в UTF-8) → сброс
        if key == ord('y') or key == ord('н'):
            M = None
            calib_points = []
            static_binary_mask = None
            try:
                os.remove(CALIB_FILE)
            except:
                pass
        else:
            # любая другая клавиша → загрузить
            if not load_calibration():
                print("Не удалось загрузить калибровку. Переход к ручной расстановке.")
                M = None
                calib_points = []

    # Создаём окно с возможностью изменения размера
    cv2.namedWindow("Robotino Control", cv2.WINDOW_NORMAL)
    # Задаем размер окна один раз при инициализации
    warp_size = (int(FIELD_WIDTH * PIX_PER_METER), int(FIELD_HEIGHT * PIX_PER_METER))
    cv2.resizeWindow("Robotino Control", warp_size[0], warp_size[1])
    cv2.setMouseCallback("Robotino Control", mouse_callback)
    
    while True:
        # ============================================
        #  РАЗДЕЛ 8.3: ЧТЕНИЕ КАДРА
        # ============================================
        
        warp_size = (int(FIELD_WIDTH * PIX_PER_METER), int(FIELD_HEIGHT * PIX_PER_METER))
        ret, frame = cap.read()
        if not ret:
            # обработка конца видео (без изменений)
            if not USE_CAMERA and LOOP_VIDEO:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, "Video ended. Close window to exit.", (50, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.imshow("Robotino Control", frame)
                while cv2.getWindowProperty("Robotino Control", cv2.WND_PROP_VISIBLE) >= 1:
                    if cv2.waitKey(100) & 0xFF == ord('q'):
                        break
                break

        # --- Шаг 1: показать калибровочные точки на ИСХОДНОМ кадре (для отладки) ---
        debug_frame = frame.copy()
        draw_calibration_points(debug_frame)
        draw_field(debug_frame)
        cv2.putText(debug_frame, "Click corners: TL, TR, BR, BL", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # --- Шаг 2: трансформация, если калибровка завершена ---
        if M is not None:
            working_frame = cv2.warpPerspective(frame, M, warp_size)
            # Без проверки чёрного
        else:
            working_frame = frame
            
        # ============================================
        #  РАЗДЕЛ 8.4: ДЕТЕКЦИЯ РОБОТА
        # ============================================
        
        det = detect_robot(working_frame)
        robot_pix = None
        marker_corners = None
        if det is not None:
            px, py, phi, corners = det        
            robot_pix = (int(px), int(py))
            marker_corners = corners
            phi = 0.0  # Голономный робот: угол всегда 0
            if M is not None:
                # Мировая X (вниз) = пиксельная Y
                wx = py / PIX_PER_METER
                # Мировая Y (вправо) = пиксельная X
                wy = px / PIX_PER_METER
                robot_pos = (wx, wy, phi)
                trajectory_world.append((wx, wy))
                trajectory_time.append(time.time())
            else:
                robot_pos = (px, py, phi)
        else:
            robot_pos = None
            
        # ============================================
        #  РАЗДЕЛ 8.5: ЧТЕНИЕ ОДОМЕТРИИ
        # ============================================
        if time.time() - last_odom_time > 0.2:
            odom_data = get_odometry()
            last_odom_time = time.time()
        else:
            odom_data = None
        
        if odom_data is not None:
            # odom_data = [x, y, phi, ...] – координаты в системе робота
            odom_x, odom_y, odom_phi = odom_data[0], odom_data[1], odom_data[2]
            odom_text = f"Odom: x={odom_x:.2f} y={odom_y:.2f} phi={math.degrees(odom_phi):.0f}"
        else:
            odom_text = ""
            
        # ============================================
        #  РАЗДЕЛ 8.6: ОБРАБОТКА ПРЕПЯТСТВИЙ
        # ============================================
        expanded_mask = None
        planning_mask = None
        if M is not None:
            if OBSTACLE_MODE == "static":
                if static_binary_mask is None:
                    static_binary_mask = detect_obstacles(working_frame, robot_pix, marker_corners, OBSTACLE_PROCESS_SCALE)
                last_binary_obs = static_binary_mask
            else:  # dynamic
                if frame_count % PROCESS_EVERY_N_FRAMES == 0:
                    last_binary_obs = detect_obstacles(working_frame, robot_pix, marker_corners, OBSTACLE_PROCESS_SCALE)
                frame_count += 1

            if last_binary_obs is not None:
                # Чистая расширенная маска для отрисовки и избегания (без рамки)
                expanded_mask = compute_expanded_mask(last_binary_obs)

                # Планировочная маска: сначала копируем expanded_mask,
                # затем дополнительно расширяем на PLANNING_EXTRA_RADIUS_M
                extra_px = int(PLANNING_EXTRA_RADIUS_M * PIX_PER_METER)
                if extra_px > 0:
                    kernel_extra = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                             (extra_px * 2 + 1, extra_px * 2 + 1))
                    planning_mask = cv2.dilate(expanded_mask, kernel_extra, iterations=1)
                else:
                    planning_mask = expanded_mask.copy()

                # Добавляем граничную рамку (только в planning_mask)
                border_px = int(BORDER_MARGIN_M * PIX_PER_METER)
                if border_px > 0:
                    planning_mask[:border_px, :] = 255
                    planning_mask[-border_px:, :] = 255
                    planning_mask[:, :border_px] = 255
                    planning_mask[:, -border_px:] = 255
            else:
                expanded_mask = None
                planning_mask = None

            draw_obstacles(working_frame, last_binary_obs, expanded_mask)
            draw_border(working_frame)
        
        # ============================================
        #  РАЗДЕЛ 8.7: ПЛАНИРОВАНИЕ МАРШРУТА
        # ============================================
        if M is not None and target_world is not None and robot_pix is not None:
            tpx = int(target_world[1] * PIX_PER_METER)
            tpy = int(target_world[0] * PIX_PER_METER)
            # Пересчитываем путь при новой цели или если маска обновилась (dynamic) и мы не достигли цели
            need_new_plan = (target_world != goal_prev)
            if OBSTACLE_MODE == "dynamic" and last_binary_obs is not None:
                need_new_plan = need_new_plan or (frame_count % PROCESS_EVERY_N_FRAMES == 0)
            if need_new_plan:
                if expanded_mask is not None:
                    planned_path = plan_path(planning_mask, robot_pix, (tpx, tpy), OBSTACLE_PROCESS_SCALE)
                    if PATH_SMOOTHING and planned_path:
                        if SIMPLIFY_EPSILON_PX > 0:
                            planned_path = simplify_path_dp(planned_path, SIMPLIFY_EPSILON_PX)
                    planned_path = smooth_path_catmull_rom(planned_path, SPLINE_RESOLUTION, PIX_PER_METER)
                    # --- сохраняем исходный план, если ещё не сохранён ---
                    if not initial_planned_path:
                        initial_planned_path = list(planned_path)
                    # ----------------------------------------------------
                    last_wp_index = 0
                else:
                    planned_path = []
                goal_prev = target_world
                waypoint_index = 0
        else:
            planned_path = []
            waypoint_index = 0
            
        # ============================================
        #  РАЗДЕЛ 8.8: ОТРИСОВКА ВСЕХ ЭЛЕМЕНТОВ
        # ============================================
        
        if M is None:
            # Исходный режим калибровки
            draw_field(working_frame)
            draw_calibration_points(working_frame)
        else:
            draw_trajectory(working_frame)
            # Вычисляем пиксельные координаты робота один раз
            if robot_pos is not None:
                rpx = int(robot_pos[1] * PIX_PER_METER)
                rpy = int(robot_pos[0] * PIX_PER_METER)
                cv2.drawMarker(working_frame, (rpx, rpy), (0, 255, 255), cv2.MARKER_CROSS, 20, 2)
                # направление
                phi = robot_pos[2]
                dx = int(30 * math.cos(phi))
                dy = int(30 * math.sin(phi))
                cv2.arrowedLine(working_frame, (rpx, rpy), (rpx+dx, rpy+dy), (0, 255, 255), 2)
                cv2.putText(working_frame, 
                            f"R({robot_pos[0]:.2f}, {robot_pos[1]:.2f}, {math.degrees(phi):.0f})",
                            (rpx+10, rpy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
            if target_world is not None:
                tpx = int(target_world[1] * PIX_PER_METER)
                tpy = int(target_world[0] * PIX_PER_METER)
                cv2.circle(working_frame, (tpx, tpy), 8, (0, 0, 255), -1)
                cv2.circle(working_frame, (tpx, tpy), 10, (0, 0, 255), 2)
                cv2.putText(working_frame, f"G({target_world[0]:.2f}, {target_world[1]:.2f})",
                            (tpx+10, tpy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

        # Отрисовка запланированного пути
        if planned_path and len(planned_path) > 1:
            pts = [(int(p[0]), int(p[1])) for p in planned_path]
            for i in range(1, len(pts)):
                cv2.line(working_frame, pts[i-1], pts[i], (255, 0, 0), 2)  # красный путь
                
            # --- ВЫБОР ЦЕЛЕВОЙ ТОЧКИ (waypoint) для отрисовки и управления ---
            tx, ty = target_world[0], target_world[1] # По умолчанию цель - финальная точка
            target_is_goal = True  # Флаг: целимся в финальную точку или в промежуточный waypoint
            if CONTROL and robot_pos is not None:
                wx, wy, _ = robot_pos
                gx, gy = target_world
                lookahead_m = WP_THRESHOLD_M
                
                found = False
                for i in range(last_wp_index, len(planned_path)):
                    wp = planned_path[i]
                    wpx = wp[1] / PIX_PER_METER
                    wpy = wp[0] / PIX_PER_METER
                    if math.hypot(wpx - wx, wpy - wy) >= lookahead_m:
                        tx, ty = wpx, wpy
                        last_wp_index = i
                        found = True
                        # Если это последний waypoint, то это финальная цель
                        target_is_goal = (i == len(planned_path) - 1)
                        break
                if not found:
                    tx, ty = gx, gy
                    target_is_goal = True
            # ------------------------------------------------------------------

            # Отрисовка текущего waypoint
            if CONTROL and robot_pos is not None:
                px_target = int(ty * PIX_PER_METER)   # мировая Y -> пиксельная X
                py_target = int(tx * PIX_PER_METER)   # мировая X -> пиксельная Y
                cv2.circle(working_frame, (px_target, py_target), 6, (255, 255, 0), -1)
            
        # Отображение одометрии на кадре
        if odom_text:
            cv2.putText(working_frame, odom_text, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1) 

         # Отображение текста слева сверху на кадре
        if M is None:
            mode_text = "Calibration"
        else:
            if CONTROL:
                mode_text = f"Control [{PATH_ALGORITHM}] ({OBSTACLE_MODE})"
            else:
                mode_text = "Video only"
        cv2.putText(working_frame, mode_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # Отображение статуса (скорость, позиция, цель) справа сверху
        draw_status_info(working_frame, robot_pos, target_world, current_vx_world, current_vy_world)
        
        cv2.imshow("Robotino Control", working_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or cv2.getWindowProperty("Robotino Control", cv2.WND_PROP_VISIBLE) < 1:
            break

        # ============================================
        #  РАЗДЕЛ 8.9: УПРАВЛЕНИЕ РОБОТОМ
        # ============================================

        if CONTROL:
            if robot_pos is not None and target_world is not None and M is not None:
                wx, wy, phi = robot_pos
                gx, gy = target_world
                distance = math.hypot(gx - wx, gy - wy)

                # ========== Динамический выход из препятствия (локальный поиск) ==========
                if OBSTACLE_MODE == "dynamic" and expanded_mask is not None:
                    rpx = int(wy * PIX_PER_METER)   # мировые Y -> пиксельные X
                    rpy = int(wx * PIX_PER_METER)   # мировые X -> пиксельные Y
                    h_mask, w_mask = expanded_mask.shape
                    if 0 <= rpx < w_mask and 0 <= rpy < h_mask and expanded_mask[rpy, rpx] != 0:
                        # Ищем ближайший свободный пиксель в квадратной окрестности
                        best_d2 = float('inf')
                        best_x = best_y = 0
                        search_radius = int(OBSTACLE_SAFE_RADIUS_M * PIX_PER_METER) + 0  # можно менять
                        for dy in range(-search_radius, search_radius + 1):
                            for dx in range(-search_radius, search_radius + 1):
                                nx = rpx + dx
                                ny = rpy + dy
                                if 0 <= nx < w_mask and 0 <= ny < h_mask and expanded_mask[ny, nx] == 0:
                                    d2 = dx*dx + dy*dy
                                    if d2 < best_d2:
                                        best_d2 = d2
                                        best_x, best_y = nx, ny
                        if best_d2 != float('inf'):
                            escape_px, escape_py = best_x, best_y
                            escape_wx = escape_py / PIX_PER_METER
                            escape_wy = escape_px / PIX_PER_METER
                            dx = escape_wx - wx
                            dy = escape_wy - wy
                            dist_escape = math.hypot(dx, dy)
                            if dist_escape > 0:
                                vx_world = (dx / dist_escape) * MAX_SPEED * 0.75
                                vy_world = (dy / dist_escape) * MAX_SPEED * 0.75
                                if time.time() - last_send_time > SEND_INTERVAL:
                                    safe_send(vx_world, vy_world, 0.0)
                                    last_send_time = time.time()
                            # Опционально: отрисовка точки выхода (синий кружок)
                            cv2.circle(working_frame, (escape_px, escape_py), 5, (255, 0, 0), -1)
                            continue  # пропускаем основную логику движения до следующего кадра
                # ============================================================

                if distance < GOAL_TOLERANCE:
                    current_vx_world = 0.0
                    current_vy_world = 0.0
                    if time.time() - last_send_time > SEND_INTERVAL:
                        safe_send(0.0, 0.0, 0.0)
                    # Сохраняем график, если ещё не сохранили для этой цели
                        if not plot_saved and (trajectory_world and trajectory_time):
                            save_trajectory_plot(
                                trajectory_world, trajectory_time, target_world,
                                initial_planned_path if initial_planned_path else planned_path,  
                                PIX_PER_METER, PATH_ALGORITHM, PATH_SMOOTHING)
                            plot_saved = True
                        last_send_time = time.time()
                else:
                    # Если путь не построен или слишком короткий – робот стоит
                    if not planned_path or len(planned_path) < 2:
                        current_vx_world = 0.0
                        current_vy_world = 0.0
                        if time.time() - last_send_time > SEND_INTERVAL:
                            safe_send(0.0, 0.0, 0.0)
                            last_send_time = time.time()
                        continue   # пропускаем всю дальнейшую логику до следующего кадра

                    # Вычисляем вектор до выбранной точки
                    dx = tx - wx
                    dy = ty - wy
                    dist_to_target = math.hypot(dx, dy)

                    # Зона начала плавного торможения (например, двойной радиус цели)
                    BRAKE_DISTANCE = GOAL_TOLERANCE * 2.0

                    # Остановка только у финальной цели
                    if target_is_goal and dist_to_target < GOAL_TOLERANCE:
                        current_vx_world = 0.0
                        current_vy_world = 0.0
                        if time.time() - last_send_time > SEND_INTERVAL:
                            safe_send(0.0, 0.0, 0.0)
                            last_send_time = time.time()
                    else:
                        # Направление к цели
                        if dist_to_target > 0:
                            # Плавное торможение при приближении к финальной цели
                            if target_is_goal and dist_to_target < BRAKE_DISTANCE:
                                # Линейно уменьшаем скорость от MAX_SPEED до минимального порога
                                speed = MAX_SPEED * (dist_to_target / BRAKE_DISTANCE)
                                # Защита от слишком малой скорости, чтобы робот не "застрял" из-за трения
                                speed = max(speed, 0.01) 
                            else:
                                speed = MAX_SPEED
                                
                            vx_world = (dx / dist_to_target) * speed
                            vy_world = (dy / dist_to_target) * speed
                        else:
                            vx_world = 0.0
                            vy_world = 0.0

                        current_vx_world = vx_world
                        current_vy_world = vy_world

                        # --- Голономный робот: скорости напрямую в мировую систему ---
                        vx_local = vx_world
                        vy_local = vy_world
                        omega = 0.0

                        if time.time() - last_send_time > SEND_INTERVAL:
                            safe_send(vx_local, vy_local, omega)
                            last_send_time = time.time()
                    
            elif robot_pos is None and target_world is not None:
                if time.time() - last_send_time > SEND_INTERVAL:
                    safe_send(0.0, 0.0, 0.0)
                    last_send_time = time.time()

    if CONTROL:
        safe_send(0.0, 0.0, 0.0)
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()