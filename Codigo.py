import cv2 
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import datetime
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
IMAGE_DIR = os.path.join(BASE_DIR, "images")

PLANTAS = ["Planta 1", "Planta 2", "Planta 3", "Planta 4"]

if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)

class PlantApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Plant Growth Tracker")

        self.plant_var = tk.StringVar(value=PLANTAS[0])
        self.plant_selector = tk.OptionMenu(root, self.plant_var, *PLANTAS)
        self.plant_selector.pack()

        self.btn_load = tk.Button(root, text="Seleccionar Imagen", command=self.load_image)
        self.btn_load.pack()

        self.canvas = tk.Label(root)
        self.canvas.pack()

        self.btn_process = tk.Button(root, text="Medir Plantas", command=self.process_image)
        self.btn_process.pack()

        self.height_label = tk.Label(root, text="Altura: ---")
        self.height_label.pack()

        self.btn_graph = tk.Button(root, text="Ver Crecimiento", command=self.show_growth)
        self.btn_graph.pack()

    def get_csv_file(self):
        planta = self.plant_var.get().replace(" ", "_").lower()
        return os.path.join(BASE_DIR, f"growth_data_{planta}.csv")

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Imágenes", "*.png;*.jpg;*.jpeg")])
        if file_path:
            self.image_path = file_path
            img = Image.open(file_path)
            img = self.resize_image(img, 300)
            self.img_tk = ImageTk.PhotoImage(img)
            self.canvas.config(image=self.img_tk)

    def resize_image(self, img, max_size):
        width, height = img.size
        ratio = min(max_size / width, max_size / height)
        new_size = (int(width * ratio), int(height * ratio))
        return img.resize(new_size, Image.LANCZOS)

    def process_image(self):
        if hasattr(self, 'image_path'):
            image = cv2.imread(self.image_path)
            if image is None:
                return

            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            lower_green = np.array([35, 50, 50])
            upper_green = np.array([90, 255, 255])
            mask_green = cv2.inRange(hsv, lower_green, upper_green)

            kernel = np.ones((3, 3), np.uint8)
            mask_clean = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel, iterations=2)

            contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            heights = []
            image_contours = image.copy()

            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if h > 10:
                    heights.append(h)
                    cv2.rectangle(image_contours, (x, y), (x + w, y + h), (255, 0, 0), 2)

            lower_coin = np.array([15, 100, 100])
            upper_coin = np.array([30, 255, 255])
            mask_coin = cv2.inRange(hsv, lower_coin, upper_coin)

            contours_coin, _ = cv2.findContours(mask_coin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            coin_height = None
            for contour in contours_coin:
                x, y, w, h = cv2.boundingRect(contour)
                if 30 < w < 100 and 30 < h < 100:
                    coin_height = h
                    cv2.rectangle(image_contours, (x, y), (x + w, y + h), (0, 255, 255), 2)

            if heights:
                avg_height_px = sum(heights) / len(heights)

                if coin_height:
                    pixel_to_cm = 2.35 / coin_height
                    avg_height_cm = avg_height_px * pixel_to_cm
                    height_text = f"Altura: {avg_height_cm:.2f} cm"
                else:
                    avg_height_cm = None
                    height_text = f"Altura: {avg_height_px:.2f} píxeles"

                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                detected_path = os.path.join(IMAGE_DIR, f"detected_{timestamp}.png")
        
                if not os.path.exists(IMAGE_DIR):
                    os.makedirs(IMAGE_DIR)
                saved = cv2.imwrite(detected_path, image_contours)
                if not saved or not os.path.exists(detected_path):
                    messagebox.showerror("Error", f"No se pudo guardar la imagen procesada en {detected_path}")
                    return

                self.show_processed_image(detected_path)
                self.height_label.config(text=height_text)
                self.save_growth_data(avg_height_cm if coin_height else avg_height_px, coin_height is not None, detected_path)

    def show_processed_image(self, path):
        img = Image.open(path)
        img = self.resize_image(img, 300)
        self.img_tk = ImageTk.PhotoImage(img)
        self.canvas.config(image=self.img_tk)

    def save_growth_data(self, avg_height, in_cm, image_path):
        unit = "cm" if in_cm else "px"
        csv_file = self.get_csv_file()
        if os.path.exists(csv_file) and os.stat(csv_file).st_size > 0:
            df = pd.read_csv(csv_file)
        else:
            df = pd.DataFrame(columns=["Fecha", "Altura", "Unidad", "Imagen"])

        new_data = pd.DataFrame([{
            "Fecha": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Altura": avg_height,
            "Unidad": unit,
            "Imagen": image_path
        }])
        df = pd.concat([df, new_data], ignore_index=True)
        df.to_csv(csv_file, index=False)

    def show_growth(self):
        csv_file = self.get_csv_file()
        if not os.path.exists(csv_file) or os.stat(csv_file).st_size == 0:
            messagebox.showinfo("Sin datos", f"No hay datos para {self.plant_var.get()}")
            return

        df = pd.read_csv(csv_file)
        if df.empty:
            messagebox.showinfo("Sin datos", f"No hay datos para {self.plant_var.get()}")
            return

        df["Fecha"] = pd.to_datetime(df["Fecha"])
        df = df.sort_values(by="Fecha")

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(df["Fecha"], df["Altura"], marker='o', linestyle='-')

        y_min, y_max = ax.get_ylim()
        offset = (y_max - y_min) * 0.05

        for i, row in df.iterrows():
            img_path = row["Imagen"]
            if os.path.exists(img_path):
                img = Image.open(img_path)
                max_dim = 100
                w, h = img.size
                scale = max_dim / max(w, h)
                img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
                img = np.array(img)
                imagebox = OffsetImage(img, zoom=1)
                y_pos = row["Altura"]
                if y_pos - offset < y_min:
                    y_pos += offset * 2
                ab = AnnotationBbox(imagebox, (row["Fecha"], y_pos), frameon=False)
                ax.add_artist(ab)

        ax.set_xlabel("Fecha y Hora")
        ax.set_ylabel(f"Altura ({df['Unidad'].iloc[0]})")
        ax.set_title(f"Crecimiento de la {self.plant_var.get()}")
        plt.xticks(rotation=45)
        plt.grid()
        plt.show()

if __name__ == "__main__":
    root = tk.Tk()
    app = PlantApp(root)
    root.mainloop()
