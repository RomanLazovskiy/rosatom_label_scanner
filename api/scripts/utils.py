import numpy as np
import cv2

def cylindrical_unwrap(image, center, radius, output_height=None):
    """
    Unwrap a circular region in an image into a rectangular strip using cv2.remap().

    :param image: Input image with a circular part.
    :param center: (x, y) coordinates of the circle's center.
    :param radius: Radius of the circular part.
    :param output_height: Desired height of the unwrapped image. Defaults to radius if not provided.
    :return: Unwrapped rectangular image.
    """
    if output_height is None:
        output_height = radius

    output_width = int(2 * np.pi * radius)

    # Create meshgrid for the output image
    theta = np.linspace(0, 2 * np.pi, output_width)
    h = np.linspace(0, radius, output_height)
    theta, h = np.meshgrid(theta, h)

    # Convert polar coordinates to Cartesian coordinates
    x_map = (center[0] + (radius - h) * np.cos(theta)).astype(np.float32)
    y_map = (center[1] + (radius - h) * np.sin(theta)).astype(np.float32)

    # Use cv2.remap to apply the mapping
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
    # Calculate the width and height of the bounding box
    width = x_max - x_min
    height = y_max - y_min

    # Determine the smallest side
    side_length = min(width, height)

    # Calculate the center of the original bounding box
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2

    # Adjust the center coordinate for the side that was larger
    if width > height:
        # Adjust the x_center to keep the new bounding box centered within the original width range
        x_center = (x_min + x_max - side_length/4) / 2
    elif height > width:
        # Adjust the y_center to keep the new bounding box centered within the original height range
        y_center = (y_min + y_max - side_length/4) / 2

    # Return the corrected center coordinates and the adjusted side length
    return x_center, y_center, side_length