import cv2
import numpy as np
from paddleocr import PaddleOCR
from PIL import Image, ImageEnhance
from io import BytesIO

class OCRProcessor:
    def __init__(self):
        self.ocr = PaddleOCR(lang='es')  # Cambia el idioma según sea necesario

    def preprocess_image(self, image_bytes):
        """
        Preprocesa una imagen desde bytes.
        """
        image = Image.open(BytesIO(image_bytes)).convert("RGB")

        # Convertir a escala de grises
        image = image.convert("L")

        # Aumentar el contraste
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(2)  # Ajusta el nivel según sea necesario
        image_np = np.array(image)
        return image_np

    def ocr_process(self, image_bytes):
        """
        Realiza OCR en una imagen proporcionada como bytes.
        """
        # Preprocesar la imagen
        processed_image = self.preprocess_image(image_bytes)
        # Convertir la imagen procesada a un archivo temporal en formato requerido por PaddleOCR
        #processed_image.save(temp_path)

        # Procesar la imagen con PaddleOCR
        results = self.ocr.ocr(processed_image)

        # Imprimir los resultados
        extracted_text = " ".join([line[1][0] for line in results[0]])
        #for line in results[0]:
        #    print(f"Texto: {line[1][0]} | Confianza: {line[1][1]}")

        return extracted_text

if __name__ == '__main__':
    # Simulación: Lee una imagen como bytes
    with open("temp/DNI_Enrique_Delgado.png", "rb") as f:  # Cambia por el archivo real que quieras usar
        image_bytes = f.read()

    # Procesar la imagen con OCRProcessor
    processor = OCRProcessor()
    text = processor.ocr_process(image_bytes)

    print("Texto extraído:", text)
