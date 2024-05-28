import cv2
import numpy as np
from tensorflow.keras.models import load_model
from data import class_names, recommendations

# Загружаем обученную модель
model = load_model('road_sings_detection.h5')

# Загружаем видеофайл
video_path = 'Вождение от первого лица - Audi A5 2.0T Coupe.mp4'  #путь к вашему видеофайлу

# Создаем объект VideoCapture для загрузки видео
cap = cv2.VideoCapture(video_path)

paused = False

while True:
    if not paused:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Обрабатываем каждый кадр
        image = cv2.resize(frame, (48, 48))
        image = np.expand_dims(image, axis=0) / 255.0
        
        # Предсказываем класс знака
        prediction = model.predict(image)
        predicted_class = np.argmax(prediction)
        
        # Получаем название знака и пояснение из словаря
        sign_info = class_names.get(predicted_class, ("Неизвестный знак", "Нет информации о знаке."))
        
        # Получаем рекомендацию из словаря
        recommendation = recommendations.get(predicted_class, "Нет рекомендаций для данного знака.")
        
        # Отображаем результат на кадре
        sign_name, sign_description = sign_info
        cv2.putText(frame, f"Предсказанный класс: {sign_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, sign_description, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Рекомендация: {recommendation}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Отображаем кадр с результатами
        cv2.imshow('Road Sign Detection', frame)
    
    # Слушаем события нажатия клавиш
    key = cv2.waitKey(1)
    
    if key == ord('q'):  # Выход из программы при нажатии клавиши 'q'
        break
    elif key == ord('p'):  # Пауза/продолжение видео при нажатии клавиши 'p'
        paused = not paused
    elif key == ord('r'):  # Перемотка видео в начало при нажатии клавиши 'r'
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    elif key == ord('f'):  # Перемотка видео вперед на 5 секунд при нажатии клавиши 'f'
        current_time_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        cap.set(cv2.CAP_PROP_POS_MSEC, current_time_ms + 5000)

# Освобождаем ресурсы
cap.release()
cv2.destroyAllWindows()
