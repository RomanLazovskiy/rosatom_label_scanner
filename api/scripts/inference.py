from ultralytics import YOLO
from transformers import pipeline, CLIPProcessor, CLIPModel
from typing import Literal
import torch
import numpy as np
import textdistance
from PIL import Image
import easyocr
import pandas as pd
from torch import nn
from torchvision import transforms
from scripts.utils import cylindrical_unwrap, adjust_bounding_box_with_center_correction


class CLIPRotationClassifier(nn.Module):
    """
    Класс для классификации изображений по углу поворота с использованием модели CLIP.

    Атрибуты
    --------
    clip_model : CLIPModel
        Модель CLIP для извлечения признаков изображения.
    fc : nn.Linear
        Линейный слой для классификации по 4 углам поворота.
    rotations : list[int]
        Список углов поворота, которые может предсказать модель.
    transform : transforms.Compose
        Трансформация изображения для подачи в модель.
    device : str
        Устройство для выполнения вычислений (например, 'cuda' или 'cpu').

    Методы
    -------
    forward(pixel_values):
        Прямой проход через модель для получения прогнозов.
    predict_rotation(image_pil):
        Прогнозирует угол поворота изображения и возвращает его с правильной ориентацией.
    """
    def __init__(self, num_classes=4, device='cuda'):
        """
        Конструктор класса для инициализации модели CLIP и классификатора поворота.

        Параметры
        ----------
        num_classes : int, optional
            Количество классов (по умолчанию 4 для углов: 0, 90, 180, 270).
        device : str, optional
            Устройство для выполнения вычислений ('cuda' или 'cpu').
        """
        super(CLIPRotationClassifier, self).__init__()
        self.clip_model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
        self.fc = nn.Linear(self.clip_model.visual_projection.out_features, num_classes)
        self.rotations = [0, 90, 180, 270]

        def preprocess(image):
            return processor(images=image, return_tensors="pt")['pixel_values'].squeeze()

        processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            preprocess
        ])
        self.transform = transform
        self.device = device

    def forward(self, pixel_values):
        """
        Прямой проход через модель для получения предсказаний.

        Параметры
        ----------
        pixel_values : torch.Tensor
            Тензор изображений, подаваемый в модель для извлечения признаков.

        Возвращает
        -------
        torch.Tensor
            Логиты для классификации углов поворота.
        """
        image_features = self.clip_model.get_image_features(pixel_values=pixel_values)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logits = self.fc(image_features)
        return logits

    def predict_rotation(self, image_pil):
        """
        Прогнозирует угол поворота изображения и возвращает изображение с правильной ориентацией.

        Параметры
        ----------
        image_pil : PIL.Image
            Изображение для прогнозирования угла поворота.

        Возвращает
        -------
        PIL.Image
            Изображение с правильной ориентацией.
        """
        image = image_pil.convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        self.eval()
        with torch.no_grad():
            logits = self(image_tensor)
            _, predicted = torch.max(logits, 1)
            predicted_rotation = self.rotations[predicted.item()]
            rotation_needed = (360 - predicted_rotation) % 360
        img = image.rotate(rotation_needed)
        return img


class MarkingOCR:
    """
    Класс для распознавания и классификации маркировок с использованием YOLO, OCR и модели поворота.

    Атрибуты
    --------
    db : pandas.DataFrame
        База данных для поиска наилучшего совпадения по тексту.
    rotation_model_clf : CLIPRotationClassifier
        Модель для классификации угла поворота.
    device : str
        Устройство для выполнения вычислений.
    detect_model : YOLO
        Модель для детекции объектов на изображении.
    ocr_type : Literal['simple', 'efficient']
        Тип используемой OCR модели: 'simple' для легкой модели, 'efficient' для более тяжелой.
    ocr_model : callable
        Модель OCR для распознавания текста на изображениях.
    circle_corection : bool
        Если True, то будет использоваться коррекция кругов.

    Методы
    -------
    _device_detect():
        Определяет доступное устройство для вычислений.
    post_detect_processing(yolo_predict):
        Обрабатывает результат предсказания YOLO.
    find_best_match(text):
        Находит наилучшее совпадение для текста в базе данных.
    detect(images_list):
        Детектирует объекты на изображениях с использованием модели YOLO.
    ocr(image_with_texts):
        Применяет OCR к изображениям и извлекает текст.
    recognition(images):
        Основной метод для обработки изображений, включая детекцию, OCR и классификацию.
    __callable__(images):
        Упрощенный интерфейс для вызова метода recognition.
    """
    def __init__(self, yolo_model_path: str,
                 db: list[str],
                 rotation_model_clf: CLIPRotationClassifier,
                 device: str = None,
                 ocr_type: Literal['simple', 'efficient'] = 'efficient',
                 path_to_efficient_ocr_model=None,
                 circle_corection: bool = True
                 ):
        """
        Конструктор класса MarkingOCR для инициализации всех необходимых моделей.

        Параметры
        ----------
        yolo_model_path : str
            Путь к модели YOLO.
        db : list[str]
            База данных для поиска наилучших совпадений.
        rotation_model_clf : CLIPRotationClassifier
            Модель для классификации углов поворота.
        device : str, optional
            Устройство для выполнения вычислений ('cuda' или 'cpu').
        ocr_type : {'simple', 'efficient'}, optional
            Тип OCR модели.
        path_to_efficient_ocr_model : str, optional
            Путь к модели эффективного OCR.
        circle_corection : bool, optional
            Если True, то будет использоваться коррекция кругов.
        """
        assert ocr_type in ['simple', 'efficient'], f'unknown ocr_type type: {ocr_type}\n available options: [simple, efficient]'

        self.db = db
        self.rotation_model_clf = rotation_model_clf
        self.device = device if device else self._device_detect()
        print(f'set device {self.device}')

        self.detect_model = YOLO(yolo_model_path)
        print('init detection model')

        self.ocr_type = ocr_type
        if self.ocr_type == 'efficient':
            model_name = path_to_efficient_ocr_model if path_to_efficient_ocr_model else "microsoft/trocr-large-stage1"
            self.ocr_model = pipeline("image-to-text", model=model_name, device=self.device)
            print('init heavy ocr model')
        elif self.ocr_type == 'simple':
            reader = easyocr.Reader(['ru'])
            self.ocr_model = reader.readtext
            print('init light ocr model')
        if circle_corection:
            self.circle_corection = circle_corection
            self.circle_detection = YOLO('yolov8m-worldv2.pt')
            self.circle_detection.set_classes(['circle object'])
            print('init circle detection model')

    def _device_detect(self):
        """
        Определяет доступное устройство для вычислений.

        Возвращает
        -------
        str
            'cuda' если доступна GPU, 'cpu' в противном случае.
        """
        if torch.cuda.is_available():
            return 'cuda'
        else:
            return 'cpu'

    @staticmethod
    def post_detect_processing(yolo_predict):
        """
        Обрабатывает результат предсказания YOLO.

        Параметры
        ----------
        yolo_predict : yolov5.Detections
            Результат предсказания модели YOLO.

        Возвращает
        -------
        np.ndarray
            Объединенное изображение с вырезанными участками.
        """
        boxes = yolo_predict.boxes  # Получаем координаты боксов
        image = yolo_predict.orig_img
        cropped_images = []

        if len(boxes) == 0:
            return image

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

    def find_best_match(self, text):
        """
        Находит наилучшее совпадение для текста в базе данных.

        Параметры
        ----------
        text : str
            Текст, для которого нужно найти наилучшее совпадение.

        Возвращает
        -------
        str
            Наилучшее совпадение для текста.
        """
        best_ratio = 0
        best_match = ''
        for _, entry in self.db.iterrows():
            ratio = textdistance.levenshtein.normalized_similarity(text, entry['ДетальАртикул'])
            if ratio > best_ratio:
                best_ratio = ratio
                best_match = entry
        return best_match['ДетальАртикул'], best_match

    def detect(self, images_list):
        """
        Детектирует объекты на изображениях с использованием модели YOLO.

        Параметры
        ----------
        images_list : list
            Список изображений для детекции.

        Возвращает
        -------
        list
            Список вырезанных изображений.
        """
        datect_result = self.detect_model(images_list, agnostic_nms=True, iou=0.8)
        crop_marking_text = [self.post_detect_processing(x) for x in datect_result]
        return crop_marking_text, datect_result

    def ocr(self, image_with_texts):
        """
        Применяет OCR к изображениям и извлекает текст.

        Параметры
        ----------
        image_with_texts : list
            Список изображений для обработки OCR.

        Возвращает
        -------
        list
            Список извлеченного текста с каждого изображения.
        """
        if self.ocr_type == 'efficient':
            if not isinstance(image_with_texts[0], Image.Image):
                image_with_texts = [Image.fromarray(x) for x in image_with_texts]
            extracted_text = self.ocr_model(image_with_texts)
            extracted_text = [x[0]['generated_text'] for x in extracted_text]
        else:
            extracted_text = [self.ocr_model(text) for text in image_with_texts]
            extracted_text = [' '.join([item[1] for item in sublist]) for sublist in extracted_text]

        return extracted_text

    def recognition(self, images: list[str], postprocessing_image: bool = False):
        """
        Основной метод для обработки изображений, включая детекцию, OCR и классификацию.

        Параметры
        ----------
        images : list
            Список изображений для обработки.
        postprocessing_image : bool, optional
            Если True, выполняется дополнительная обработка изображения (например, поворот).

        Возвращает
        -------
        list
            Список наилучших совпадений для каждого текста.
        """
        if self.circle_corection:
            circles_detect = self.circle_detection(images)
            images = []
            for predict in circles_detect:
                image = predict.orig_img
                if len(predict):
                    center_x, center_y, diametr = adjust_bounding_box_with_center_correction(
                        *predict.boxes.xyxy.tolist()[0])
                    unwrapped_image = cylindrical_unwrap(image, (int(center_x), int(center_y)), int(diametr / 2))
                    images.append(unwrapped_image)
                else:
                    images.append(image)
            print('unwrapped circle images')

        if postprocessing_image:
            preprocessed_images = [
                self.rotation_model_clf.predict_rotation(Image.open(x) if isinstance(x, str) else Image.fromarray(x))
                for x in images]
            print('rotated images')

        croped_images, _ = self.detect(preprocessed_images)
        print('detect text')

        texts = self.ocr(croped_images)
        print('OCR')
        best_match = [(self.find_best_match(text), text) for text in texts]
        return best_match

    def __call__(self, images: list[str]):
        """
        Упрощенный интерфейс для вызова метода recognition.

        Параметры
        ----------
        images : list
            Список изображений для обработки.

        Возвращает
        -------
        list
            Список наилучших совпадений для каждого текста.
        """
        return self.recognition(images)


def get_inference_class(yolo_model_path,
                        db_excel_path,
                        path_to_rotate_model,
                        path_to_efficient_ocr_model=None,
                        device=None):
    """
    Функция для создания объекта для распознавания маркировок с использованием YOLO, OCR и модели поворота.

    Параметры
    ----------
    yolo_model_path : str
        Путь к модели YOLO.
    db_excel_path : str
        Путь к файлу базы данных в формате Excel.
    path_to_rotate_model : str
        Путь к модели для классификации угла поворота.
    path_to_efficient_ocr_model : str, optional
        Путь к модели эффективного OCR.
    device : str, optional
        Устройство для выполнения вычислений ('cuda' или 'cpu').

    Возвращает
    -------
    MarkingOCR
        Экземпляр класса MarkingOCR для выполнения распознавания.
    """
    device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
    rot_model = CLIPRotationClassifier(device=device)
    rot_model.load_state_dict(torch.load(path_to_rotate_model, weights_only=True, map_location=device))
    rot_model.eval()
    rot_model.to(device)

    db = pd.read_excel(db_excel_path)
    recognitor = MarkingOCR(
        yolo_model_path=yolo_model_path,
        db=db,
        rotation_model_clf=rot_model,
        path_to_efficient_ocr_model=path_to_efficient_ocr_model,
        device=device
    )
    return recognitor
