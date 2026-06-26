import cv2

print("Проверка камер с бэкендом DirectShow:")
for i in range(5):
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    if cap.isOpened():
        print(f"  Индекс {i}: работает")
        cap.release()
    else:
        print(f"  Индекс {i}: НЕ работает")

print("\nПроверка камер с авто-бэкендом:")
for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"  Индекс {i}: работает")
        cap.release()
    else:
        print(f"  Индекс {i}: НЕ работает")