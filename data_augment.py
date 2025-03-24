import os
import cv2
import numpy as np

_count = 0

def augment_image_even_odd(image_path, destination_dir, even_suffix="0.png", odd_suffix="1.png"):
    """
    Augments an image by setting every other pixel to black, creating even and odd versions.

    Args:
        image_path (str): Path to the input image.
        destination_dir (str): Path to the directory to save the augmented images.
        even_suffix (str): Suffix for the even-pixel-blacked-out image.
        odd_suffix (str): Suffix for the odd-pixel-blacked-out image.
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not read image from {image_path}")
            return

        rows, cols, channels = img.shape
        even_augmented = img.copy()
        odd_augmented = img.copy()

        # Black out even pixels
        for i in range(rows):
            for j in range(cols):
                if (i + j) % 2 == 0:
                    even_augmented[i, j] = [0, 0, 0]  # Black

        # Black out odd pixels
        for i in range(rows):
            for j in range(cols):
                if (i + j) % 2 != 0:
                    odd_augmented[i, j] = [0, 0, 0]  # Black

        # Save augmented images
        base_filename = os.path.splitext(os.path.basename(image_path))[0]
        even_path = os.path.join(destination_dir, base_filename + even_suffix)
        odd_path = os.path.join(destination_dir, base_filename + odd_suffix)

        cv2.imwrite(even_path, even_augmented)
        cv2.imwrite(odd_path, odd_augmented)
        print(f"Augmented images saved to {destination_dir}")

    except Exception as e:
        print(f"An error occurred: {e}")

def augment_directory_even_odd(source_dir, destination_dir):
    """
    Augments all images in a source directory and saves the augmented versions to a destination directory.

    Args:
        source_dir (str): Path to the directory containing input images.
        destination_dir (str): Path to the directory to save augmented images.
    """
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    for filename in os.listdir(source_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')):
            source_path = os.path.join(source_dir, filename)
            augment_image_even_odd(source_path, destination_dir)


augment_directory_even_odd("./denoised_images","./augmented_images")