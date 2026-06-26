import cv2
import numpy as np

# ---- НАСТРОЙКА (измените индекс, если камера не 0) ----
CAMERA_INDEX = 0

def nothing(x):
    pass

def create_trackbars(window_name):
    """Создаёт трекбары с короткими английскими подписями (кириллица не поддерживается)."""
    # Black
    cv2.createTrackbar('Blk V', window_name, 150, 255, nothing)

    # Red1
    cv2.createTrackbar('R1 H l', window_name, 0, 179, nothing)
    cv2.createTrackbar('R1 S l', window_name, 50, 255, nothing)
    cv2.createTrackbar('R1 V l', window_name, 50, 255, nothing)
    cv2.createTrackbar('R1 H h', window_name, 10, 179, nothing)
    cv2.createTrackbar('R1 S h', window_name, 255, 255, nothing)
    cv2.createTrackbar('R1 V h', window_name, 255, 255, nothing)

    # Red2
    cv2.createTrackbar('R2 H l', window_name, 160, 179, nothing)
    cv2.createTrackbar('R2 S l', window_name, 50, 255, nothing)
    cv2.createTrackbar('R2 V l', window_name, 50, 255, nothing)
    cv2.createTrackbar('R2 H h', window_name, 180, 179, nothing)
    cv2.createTrackbar('R2 S h', window_name, 255, 255, nothing)
    cv2.createTrackbar('R2 V h', window_name, 255, 255, nothing)

    # Green
    cv2.createTrackbar('Gn H l', window_name, 40, 179, nothing)
    cv2.createTrackbar('Gn S l', window_name, 50, 255, nothing)
    cv2.createTrackbar('Gn V l', window_name, 50, 255, nothing)
    cv2.createTrackbar('Gn H h', window_name, 80, 179, nothing)
    cv2.createTrackbar('Gn S h', window_name, 255, 255, nothing)
    cv2.createTrackbar('Gn V h', window_name, 255, 255, nothing)

    # Blue
    cv2.createTrackbar('Bl H l', window_name, 100, 179, nothing)
    cv2.createTrackbar('Bl S l', window_name, 50, 255, nothing)
    cv2.createTrackbar('Bl V l', window_name, 50, 255, nothing)
    cv2.createTrackbar('Bl H h', window_name, 130, 179, nothing)
    cv2.createTrackbar('Bl S h', window_name, 255, 255, nothing)
    cv2.createTrackbar('Bl V h', window_name, 255, 255, nothing)

    # Morphology
    cv2.createTrackbar('Opn', window_name, 5, 20, nothing)
    cv2.createTrackbar('Cls', window_name, 11, 20, nothing)

    # Expansion
    cv2.createTrackbar('Exp', window_name, 30, 200, nothing)


def get_masks(frame):
    """Возвращает бинарную и расширенную маски на основе текущих положений трекбаров."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Black
    black_max_v = cv2.getTrackbarPos('Blk V', 'Mask')
    black_mask = cv2.inRange(hsv, (0, 0, 0), (179, 255, black_max_v))

    # Red1
    r1_low = [cv2.getTrackbarPos(f'R1 {x} l', 'Mask') for x in ('H','S','V')]
    r1_high = [cv2.getTrackbarPos(f'R1 {x} h', 'Mask') for x in ('H','S','V')]
    red1 = cv2.inRange(hsv, np.array(r1_low), np.array(r1_high))

    # Red2
    r2_low = [cv2.getTrackbarPos(f'R2 {x} l', 'Mask') for x in ('H','S','V')]
    r2_high = [cv2.getTrackbarPos(f'R2 {x} h', 'Mask') for x in ('H','S','V')]
    red2 = cv2.inRange(hsv, np.array(r2_low), np.array(r2_high))

    # Green
    g_low = [cv2.getTrackbarPos(f'Gn {x} l', 'Mask') for x in ('H','S','V')]
    g_high = [cv2.getTrackbarPos(f'Gn {x} h', 'Mask') for x in ('H','S','V')]
    green = cv2.inRange(hsv, np.array(g_low), np.array(g_high))

    # Blue
    b_low = [cv2.getTrackbarPos(f'Bl {x} l', 'Mask') for x in ('H','S','V')]
    b_high = [cv2.getTrackbarPos(f'Bl {x} h', 'Mask') for x in ('H','S','V')]
    blue = cv2.inRange(hsv, np.array(b_low), np.array(b_high))

    # Combine all colors
    combined = cv2.bitwise_or(black_mask, red1)
    combined = cv2.bitwise_or(combined, red2)
    combined = cv2.bitwise_or(combined, green)
    combined = cv2.bitwise_or(combined, blue)

    # Morphology
    open_k  = cv2.getTrackbarPos('Opn', 'Mask')
    close_k = cv2.getTrackbarPos('Cls', 'Mask')
    if open_k > 0:
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel_open)
    if close_k > 0:
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_close)

    # Expansion
    expand_px = cv2.getTrackbarPos('Exp', 'Mask')
    if expand_px > 0:
        kernel_expand = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                  (expand_px*2+1, expand_px*2+1))
        expanded = cv2.dilate(combined, kernel_expand, iterations=1)
    else:
        expanded = combined.copy()

    return combined, expanded


def open_camera(index):
    """Пытается открыть камеру с DirectShow, затем с авто-бэкендом."""
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if cap.isOpened():
        return cap
    cap = cv2.VideoCapture(index)
    return cap


def main():
    cap = open_camera(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"Камера {CAMERA_INDEX} не открылась. Поиск доступных камер...")
        for i in range(5):
            cap = open_camera(i)
            if cap.isOpened():
                print(f"Используется камера {i}")
                break
        else:
            print("Не удалось открыть ни одну камеру")
            return

    cv2.namedWindow('Mask', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Mask', 800, 600)
    create_trackbars('Mask')

    # Подробная инструкция в консоли (русская)
    print("=" * 60)
    print("  СОКРАЩЕНИЯ В ПОЛЗУНКАХ:")
    print("  Blk V    = максимальная яркость для чёрного")
    print("  R1 H l/h = Красный1 H мин/макс")
    print("  R1 S l/h = Красный1 S мин/макс")
    print("  R1 V l/h = Красный1 V мин/макс")
    print("  R2 ...   = Красный2 (аналогично)")
    print("  Gn ...   = Зелёный")
    print("  Bl ...   = Синий")
    print("  Opn      = размер ядра открытия")
    print("  Cls      = размер ядра закрытия")
    print("  Exp      = радиус расширенной зоны (пиксели)")
    print("  Для выхода нажмите ESC.")
    print("=" * 60)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        binary, expanded = get_masks(frame)

        cv2.imshow('Original', frame)
        cv2.imshow('Binary mask', binary)
        cv2.imshow('Expanded mask', expanded)

        # Overlay with contours
        overlay = frame.copy()
        contours, _ = cv2.findContours(expanded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 255, 255), 2)
        expand_px = cv2.getTrackbarPos('Exp', 'Mask')
        open_k  = cv2.getTrackbarPos('Opn', 'Mask')
        close_k = cv2.getTrackbarPos('Cls', 'Mask')
        cv2.putText(overlay, f"Expand: {expand_px} px | Open: {open_k} | Close: {close_k}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow('Overlay', overlay)

        if cv2.waitKey(1) & 0xFF == 27:   # ESC
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()