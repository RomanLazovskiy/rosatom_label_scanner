import torch
from torch import nn
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel, pipeline
from ultralytics import YOLO
from PIL import Image
from typing import Literal, List
import easyocr
import numpy as np
import logging

from scripts.utils import device_detect, find_best_match

device = device_detect()


class CLIPRotationClassifier(nn.Module):
    def __init__(self, num_classes=4, device=device, classifier_path='models/clip_rotation_classifier.pth'):
        super(CLIPRotationClassifier, self).__init__()
        self.device = device
        self.rotations = [0, 90, 180, 270]
        self.clip_model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32').to(self.device)
        self.fc = nn.Linear(self.clip_model.visual_projection.out_features, num_classes)
        self.load_state_dict(torch.load(classifier_path, map_location=self.device), strict=False)
        self.eval()
        self.to(self.device)

        self.processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            lambda image: self.processor(images=image, return_tensors="pt")['pixel_values'].squeeze()
        ])

    def forward(self, pixel_values):
        image_features = self.clip_model.get_image_features(pixel_values=pixel_values.to(self.device))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logits = self.fc(image_features)
        return logits

    def predict_rotation(self, image_pil):
        image = image_pil.convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.forward(image_tensor)
            _, predicted = torch.max(logits, 1)
            predicted_rotation = self.rotations[predicted.item()]
            rotation_needed = (360 - predicted_rotation) % 360

        rotated_image = image.rotate(rotation_needed)
        return rotated_image


class MarkingOCR:
    def __init__(self, yolo_model_path: str,
                 db: List[str],
                 device: str = None,
                 ocr_type: Literal['simple', 'efficient'] = 'efficient'):
        assert ocr_type in ['simple',
                            'efficient'], f'Unknown ocr_type: {ocr_type}. Available options: [simple, efficient]'

        self.db = db
        self.device = device if device else device_detect()
        self.detect_model = YOLO(yolo_model_path).to(self.device)
        self.ocr_type = ocr_type

        if self.ocr_type == 'efficient':
            model_name = "microsoft/trocr-large-stage1"
            self.ocr_model = pipeline("image-to-text", model=model_name, device=self.device)
        elif self.ocr_type == 'simple':
            self.reader = easyocr.Reader(['ru'])
            self.ocr_model = self.reader.readtext

        self.rotation_classifier = CLIPRotationClassifier(device=self.device)

    @staticmethod
    def post_detect_processing(yolo_predict):
        boxes = yolo_predict.boxes  # Получаем координаты боксов
        image = yolo_predict.orig_img
        # Список для хранения вырезанных участков
        cropped_images = []

        if len(boxes) == 0:
            # logging.info("Боксов не обнаружено на изображении.")
            return image
        # Вырезание каждого бокса из изображения
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cropped_image = image[y1:y2, x1:x2]
            cropped_images.append(cropped_image)

        # Соединение вырезанных изображений последовательно
        if len(cropped_images) > 1:
            # Определяем общую ширину и высоту для объединенного изображения
            total_height = sum(img.shape[0] for img in cropped_images)
            max_width = max(img.shape[1] for img in cropped_images)

            # Создаем пустое изображение для объединения
            combined_image = np.zeros((total_height, max_width, 3), dtype=np.uint8)

            # Заполняем объединенное изображение
            y_offset = 0
            for img in cropped_images:
                height = img.shape[0]
                combined_image[y_offset:y_offset + height, :img.shape[1]] = img
                y_offset += height
        else:
            # Если только одно изображение, просто используем его
            combined_image = cropped_images[-1]
        return combined_image

    def detect(self, images_list):
        detect_result = self.detect_model(images_list)
        crop_marking_text = [self.post_detect_processing(x) for x in detect_result]
        logging.info(f"Cropped {len(crop_marking_text)}")
        return crop_marking_text

    def ocr(self, image_with_texts):
        logging.info(f"Ocr type: {self.ocr_type}")
        logging.info(f"image_with_texts type: {type(image_with_texts[0])}")
        if self.ocr_type == 'efficient':
            if not isinstance(image_with_texts[0], Image.Image):
                image_with_texts = [Image.fromarray(x) for x in image_with_texts]
            logging.info("image_with_texts type: ", type(image_with_texts[0]))
            extracted_text = self.ocr_model(image_with_texts)
            extracted_text = [x[0]['generated_text'] for x in extracted_text]
            logging.info("extracted_text", extracted_text)
        else:
            extracted_text = [self.ocr_model(text) for text in image_with_texts]
            extracted_text = [' '.join([item[1] for item in sublist]) for sublist in extracted_text]

        return extracted_text

    def recognition(self, images: List[str], postprocessing_image: bool = False):
        logging.info(f"Recognition started for {len(images)} images with postprocessing set to {postprocessing_image}.")

        if postprocessing_image:
            preprocessed_images = [self.rotation_classifier.predict_rotation(Image.open(img_path)) for img_path in
                                   images]
            logging.info('End of image rotation processing.')

        cropped_images = self.detect(preprocessed_images)
        logging.info('End of detection.')

        texts = self.ocr(cropped_images)
        logging.info('End of OCR.')

        best_matches = [(find_best_match(text, self.db)[0], text) for text in texts]
        logging.info(f"Best matches: {best_matches}")
        return best_matches

    def __call__(self, images: List[str], postprocessing_image: bool = False):
        return self.recognition(images, postprocessing_image)
