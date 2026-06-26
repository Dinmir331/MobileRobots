# ================================================================
#  МОДУЛЬ СВЯЗИ С ROBOTINO (Robotino_communication.py)
# ================================================================

import socket
import requests
import time                     # добавлено: необходим для работы send_velocity

# Параметры подключения к роботу
IP_ADDRESS = '192.168.0.1'      # локальный IP‑адрес робота
PORT = 80                       # порт для HTTP‑запросов

# Глобальная переменная для подавления частых сообщений об ошибках
_last_error_time = 0.0

# ------------------------------------------------------------------
def connect_to_robotino():
    """Устанавливает TCP‑соединение с роботом (необязательно для HTTP)."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((IP_ADDRESS, PORT))
        print("Успешно подключились к Robotino!")
        return sock
    except Exception as e:
        print(f"Ошибка подключения к Robotino: {e}")
        return None

# ------------------------------------------------------------------
def get_odometry():
    """Запрашивает одометрию робота. Возвращает список [x, y, phi, ...] или None."""
    try:
        url = f"http://{IP_ADDRESS}/data/odometry"
        response = requests.get(url, timeout=0.2)
        if response.status_code == 200:
            odometry_readings = response.json()
            if len(odometry_readings) == 7:
                # [0] – X, [1] – Y, [2] – угол PHI
                return odometry_readings
            else:
                print("Получено неожиданное количество значений датчиков.")
        else:
            print(f"Ошибка: получен код состояния {response.status_code}")
    except Exception as e:
        print(f"Ошибка получения одометрии: {e}")
    return None

# ------------------------------------------------------------------
def get_proximity_sensor_values():
    """Запрашивает данные датчиков расстояния. Возвращает список из 9 значений или None."""
    try:
        url = f"http://{IP_ADDRESS}/data/distancesensorarray"
        response = requests.get(url, timeout=0.2)
        if response.status_code == 200:
            sensor_values = response.json()
            if len(sensor_values) == 9:
                return sensor_values
            else:
                print("Получено неожиданное количество значений датчиков.")
        else:
            print(f"Ошибка: получен код состояния {response.status_code}")
    except Exception as e:
        print(f"Ошибка получения данных с датчиков: {e}")
    return None

# ------------------------------------------------------------------
def send_velocity(vx, vy, omega):
    """Отправляет команду скоростей роботу (всенаправленное движение)."""
    global _last_error_time
    url = f"http://{IP_ADDRESS}/data/omnidrive"
    # Преобразуем в обычный float для сериализации JSON
    data = [float(vx), float(vy), float(omega)]
    try:
        response = requests.post(url, json=data, timeout=0.2)
        if response.status_code == 200:
            pass   # команда успешно принята
        else:
            print(f"Ошибка отправки данных: {response.status_code} - {response.text}")
    except Exception as e:
        # Выводим сообщение не чаще одного раза в секунду
        if time.time() - _last_error_time > 1.0:
            print(f"Ошибка отправки данных: {e}")
            _last_error_time = time.time()