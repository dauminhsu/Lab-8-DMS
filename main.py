import cv2
import numpy as np


def task1():
    # Загрузка изображения
    img = cv2.imread('./images/variant-5.jpg')

    # Добавление шума
    noise = np.zeros(img.shape, np.uint8)

    cv2.randu(noise, (0, 0, 0), (255, 255, 255))

    noise_images = {
        'noise_light': [0, 180],
        'noise_dark': [75, 255],
        'normal': [30, 220],
        'high_noise': [75, 180],
        'low_noise': [5, 250]
    }

    # Создать новые изображении
    for file_name in noise_images:
        new_img = np.copy(img)
        noise_range = noise_images[file_name]

        pepper = noise < noise_range[0]  # giam -> toi hon, tang -> sang hon
        new_img[pepper] = 0

        salt = noise > noise_range[1]  # giam -> sang hon, tang -> toi hon
        new_img[salt] = 255

        cv2.imwrite(f'./noise_images/{file_name}.png', new_img)

        # Отображение изображения
        cv2.imshow('Task 1', new_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def task2():
    # Создание объекта VideoCapture для захвата видео с камеры
    cap = cv2.VideoCapture(0)

    # Цикл обработки каждого кадра видео
    while True:
        # Получение кадра видео с камеры
        ret, frame = cap.read()
        if not ret:
            break

        # Chuyen anh co mau -> anh den trang
        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Lay do chenh lech la 180 (cang cao thi chenh lech lon moi ve diem)
        ret, thresh = cv2.threshold(img_gray, 180, 255, cv2.THRESH_BINARY)

        # tim duong vien
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_NONE)

        img_copy = frame.copy()
        img_copy = cv2.drawContours(
            img_copy, contours, -1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow('Task 2', img_copy)

        # Выход из цикла по нажатию клавиши "q"
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Освобождение ресурсов
    cap.release()
    cv2.destroyAllWindows()


def task3():
    # Создание объекта VideoCapture для захвата видео с камеры
    cap = cv2.VideoCapture(0)

    # Цикл обработки каждого кадра видео
    while True:
        # Получение кадра видео с камеры
        ret, frame = cap.read()
        if not ret:
            break

        # Chuyen anh co mau -> anh den trang
        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        ret, thresh = cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY)
        # tim duong vien
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_NONE)  # ?
        img_copy = frame.copy()

        for i, contour in enumerate(contours):
            # Xac mau sac cho vien
            color = (0, 255, 0)  # Mac dinh la mau xanh la
            for j, contour_point in enumerate(contour):
                if contour_point[0][0] < 50 and contour_point[0][1] < 50:
                    # To mau xanh lam neu nam o trai tren
                    color = (255, 0, 0)
                    break
                elif contour_point[0][0] >= img_copy.shape[1] - 50 and contour_point[0][1] >= img_copy.shape[0] - 50:
                    # To mau do neu nam o phai duoi
                    color = (0, 0, 255)
                    break

            for j, contour_point in enumerate(contour):
                cv2.circle(
                    img_copy, ((contour_point[0][0],  contour_point[0][1])), 1, color, 1, cv2.LINE_AA)

        cv2.rectangle(img_copy, (0, 0), (50, 50), (255, 0, 0), 1)
        cv2.rectangle(img_copy, (img_copy.shape[1] - 50, img_copy.shape[0] - 50),
                      (img_copy.shape[1], img_copy.shape[0]), (0, 0, 255), 1)
        cv2.imshow('Task 3', img_copy)

        # Выход из цикла по нажатию клавиши "q"
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Освобождение ресурсов
    cap.release()
    cv2.destroyAllWindows()


def task_dop():
    # Создание объекта VideoCapture для захвата видео с камеры
    cap = cv2.VideoCapture(0)

    # Цикл обработки каждого кадра видео
    while True:
        # Получение кадра видео с камеры
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)
        if not ret:
            break

        # Chuyen anh co mau -> anh den trang
        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        ret, thresh = cv2.threshold(img_gray, 180, 255, cv2.THRESH_BINARY)
        # tim duong vien
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_NONE)

        img_copy = frame.copy()
        fly = cv2.imread("./images/fly64.png", cv2.IMREAD_UNCHANGED)
        fly = cv2.resize(fly, (8, 8))

        for _, contour in enumerate(contours):
            for _, contour_point in enumerate(contour):
                x, y = contour_point[0]
                if y - fly.shape[0] // 2 >= 0 and y + fly.shape[0] // 2 < img_copy.shape[0] and x - fly.shape[1] // 2 >= 0 and x + fly.shape[1] // 2 < img_copy.shape[1]:
                    if (np.min(img_copy[y - fly.shape[0] // 2:y + fly.shape[0] // 2,
                                        x - fly.shape[1] // 2:x + fly.shape[1] // 2, 3]) == 0):  # neu da ve ruoi roi thi ko thuc hien gi
                        continue

                    # Neu chua to ruoi thi to con ruoi
                    img_copy[y - fly.shape[0] // 2:y + fly.shape[0] // 2,
                             x - fly.shape[1] // 2:x + fly.shape[1] // 2, :3] = fly[:, :, :3]

                    # Danh dau la da to
                    img_copy[y - fly.shape[0] // 2:y + fly.shape[0] // 2,
                             x - fly.shape[1] // 2:x + fly.shape[1] // 2, 3] = 0

        img_copy = img_copy[:, :, :3]
        cv2.imshow('Task Dop', img_copy)

        # Выход из цикла по нажатию клавиши "q"
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Освобождение ресурсов
    cap.release()
    cv2.destroyAllWindows()


task1()
task2()
task3()
task_dop()
