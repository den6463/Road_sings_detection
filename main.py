# main.py
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from data import class_names, recommendations  # Импортируем словари из data.py

# Загружаем обученную модель
model = load_model('road_sings_detection.h5')

class RoadSignRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Road Sign Recognition App")
        self.root.geometry("300x500")  # Устанавливаем размер окна

        self.image_label = tk.Label(self.root)
        self.image_label.pack(padx=10, pady=10)

        self.browse_button = tk.Button(self.root, text="Выбрать изображение", command=self.browse_image)
        self.browse_button.pack(pady=10)

        self.recognize_button = tk.Button(self.root, text="Распознать знак", command=self.recognize_sign)
        self.recognize_button.pack(pady=10)

        self.result_label = tk.Label(self.root, text="")
        self.result_label.pack(pady=10)

    def browse_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.display_image(file_path)

    def display_image(self, file_path):
        image = Image.open(file_path)
        image = image.resize((300, 300), Image.LANCZOS)
        photo = ImageTk.PhotoImage(image)

        self.image_label.config(image=photo)
        self.image_label.image = photo

        self.image_array_original = cv2.imread(file_path)
        self.image_array_original = cv2.cvtColor(self.image_array_original, cv2.COLOR_BGR2RGB)
        self.image_array = cv2.resize(self.image_array_original, (48, 48))
        self.image_array = np.expand_dims(self.image_array, axis=0) / 255.0

        self.result_label.config(text="")

    def recognize_sign(self):
        if hasattr(self, 'image_array'):
            # Выполняем предсказание
            prediction = model.predict(self.image_array)
            predicted_class = np.argmax(prediction)
            print(f"Предсказанный класс: {predicted_class}")

            # Получаем название знака и пояснение из словаря
            sign_info = class_names.get(predicted_class, ("Неизвестный знак", "Нет информации о знаке."))

            # Получаем рекомендацию из словаря
            recommendation = recommendations.get(predicted_class, "Нет рекомендаций для данного знака.")

            # Выводим результат в интерфейс
            sign_name, sign_description = sign_info
            self.result_label.config(text=f"Предсказанный класс: {sign_name}\n{sign_description}\n\nРекомендация: {recommendation}")

        else:
            self.result_label.config(text="Выберите изображение перед распознаванием")

if __name__ == "__main__":
    root = tk.Tk()
    app = RoadSignRecognitionApp(root)
    root.mainloop()
