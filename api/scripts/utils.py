import numpy as np
import cv2


def cylindrical_unwrap(image, center, radius, output_height=None):
    """
    Разворачивает круглый участок изображения в прямоугольную полоску с использованием функции cv2.remap().

    Параметры
    ----------
    image : np.ndarray
        Входное изображение, содержащее круглый участок.
    center : tuple
        Координаты центра круга в виде (x, y).
    radius : int
        Радиус круглого участка, который нужно развернуть.
    output_height : int, optional
        Желаемая высота развернутого изображения. По умолчанию равна радиусу.

    Возвращает
    -------
    np.ndarray
        Развёрнутое прямоугольное изображение.

    Примечания
    ---------
    Использует полярные координаты для преобразования в прямоугольные и метод cv2.remap() для выполнения преобразования.
    """
    if output_height is None:
        output_height = radius

    output_width = int(2 * np.pi * radius)

    # Создание сетки для выходного изображения
    theta = np.linspace(0, 2 * np.pi, output_width)
    h = np.linspace(0, radius, output_height)
    theta, h = np.meshgrid(theta, h)

    # Преобразование полярных координат в декартовы
    x_map = (center[0] + (radius - h) * np.cos(theta)).astype(np.float32)
    y_map = (center[1] + (radius - h) * np.sin(theta)).astype(np.float32)

    # Применение отображения с использованием cv2.remap
    unwrapped_image = cv2.remap(
        image,
        x_map,
        y_map,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    return unwrapped_image


def adjust_bounding_box_with_center_correction(x_min, y_min, x_max, y_max):
    """
    Корректирует границы ограничивающего прямоугольника с учетом исправленного центра.

    Параметры
    ----------
    x_min : int
        Координата x левой нижней вершины ограничивающего прямоугольника.
    y_min : int
        Координата y левой нижней вершины ограничивающего прямоугольника.
    x_max : int
        Координата x правой верхней вершины ограничивающего прямоугольника.
    y_max : int
        Координата y правой верхней вершины ограничивающего прямоугольника.

    Возвращает
    -------
    tuple
        Исправленные координаты центра (x_center, y_center) и длина стороны (side_length), на основе минимальной стороны исходного прямоугольника.

    Примечания
    ---------
    Возвращает центр с коррекцией для сохранения пропорций, где меньшая сторона остается фиксированной.
    """
    # Вычисление ширины и высоты ограничивающего прямоугольника
    width = x_max - x_min
    height = y_max - y_min

    # Определение меньшей стороны
    side_length = min(width, height)

    # Вычисление центра исходного прямоугольника
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2

    # Корректировка координат центра для большей стороны
    if width > height:
        # Корректировка x_center для сохранения нового прямоугольника внутри исходного диапазона по ширине
        x_center = (x_min + x_max - side_length / 4) / 2
    elif height > width:
        # Корректировка y_center для сохранения нового прямоугольника внутри исходного диапазона по высоте
        y_center = (y_min + y_max - side_length / 4) / 2

    # Возвращаем откорректированные координаты центра и новую длину стороны
    return x_center, y_center, side_length
