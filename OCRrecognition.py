import cv2
import numpy as np
from paddleocr import PaddleOCR
from PIL import Image, ImageEnhance
import glob
import os
 
class OCRProcessor(path):
    
    def __init__(self):
         self.images_files = self.detect_images(path)
         


    def _detect_images(self, temp_folder_path):

    # Buscar todos los archivos en la carpeta que son imágenes
        image_files = [
            file for file in glob.glob(os.path.join(temp_folder_path, "*")) 
            if file.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

    # Mostrar los archivos de imagen
        print("Archivos de imagen encontrados:")
        for file in image_files:
            print(file)
        return image_files


    def preprocess_image(self):
            for path in self.images_files:
                image = Image.open(path).convert("RGB")

                # Convertir a escala de grises
                image = image.convert("L")

                # Aumentar el contraste
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(2)  # Ajusta el nivel según sea necesario

                # Guardar la imagen procesada (opcional)
                image.save("documento_procesado.png")

        
    def ocr(path):
    # Inicializar OCR
        
        ocr = PaddleOCR(lang='es')  # Cambia el idioma según sea necesario

    # Procesar una imagen
        results = ocr.ocr(path)

    # Imprimir los resultados
        for line in results[0]:
            print("Texto:", line[1][0], "| Confianza:", line[1][1])
        return " ".join([line[1][0] for line in results[0]])


if __name__ =='__main__':
    path = 'temp/'

