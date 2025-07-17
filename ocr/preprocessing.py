"""Preprocessing pipeline for OCR: deskew, denoise, threshold."""

import os
import cv2
import numpy as np
from logger import get_logger
import time

logger = get_logger(__name__)


def preprocess_image(image_path: str, output_folder: str = "data/primary"):
    """
    Preprocess an image: deskew, denoise, threshold, and save to data/primary folder.

    Parameters
    ----------
    image_path : str
        Path to original image file.
    output_folder : str, optional
        Folder to save processed image.

    Returns
    -------
    str
        Path to preprocessed image file.
    """
    logger.info(f"Starting preprocessing for {image_path}")
    start_time = time.time()
    try:
        os.makedirs(output_folder, exist_ok=True)
        filename = os.path.basename(image_path)

        # Read image
        image = cv2.imread(image_path)
        cv2.imwrite(os.path.join(output_folder, f"0raw_{filename}"), image)

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        logger.debug("Image converted to grayscale")

        # Bilateral filter as main preprocessing
        bilateral = cv2.bilateralFilter(
            gray, d=3, sigmaColor=30, sigmaSpace=30
        )
        cv2.imwrite(
            os.path.join(output_folder, f"1bilateral_{filename}"), bilateral
        )
        logger.debug("Applied bilateral filter")

        # Deskew image
        coords = np.column_stack(np.where(bilateral > 0))
        angle = cv2.minAreaRect(coords)[-1]
        if -45 < angle < 45:
            angle = -angle
        else:
            angle = 0
        logger.debug(f"Computed rotation angle: {angle:.2f} degrees")

        (h, w) = bilateral.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        deskewed = cv2.warpAffine(
            bilateral,
            rotation_matrix,
            (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )
        logger.debug("Deskewed image generated")

        # Save processed image only if in development environment
        env = os.getenv("ENVIRONMENT", "development")
        if env == "development":
            filename = os.path.basename(image_path)
            processed_path = os.path.join(
                output_folder, f"preprocessed_{filename}"
            )
            cv2.imwrite(processed_path, deskewed)

        elapsed = time.time() - start_time
        logger.info(
            f"Completed preprocessing for {image_path} in {elapsed:.2f} seconds"
        )

        return deskewed

    except Exception:
        logger.exception(
            f"Exception occurred during preprocessing for {image_path}"
        )
        raise
