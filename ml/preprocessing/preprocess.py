import cv2
import numpy as np
from typing import Tuple

def resize_image(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    return cv2.resize(image, size)

def normalize_image(image: np.ndarray) -> np.ndarray:
    return image.astype(np.float32) / 255.0

def denormalize_image(image: np.ndarray, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)) -> np.ndarray:
    img = image.copy()
    if img.ndim == 3 and img.shape[0] == 3:  # CHW format
        for c in range(3):
            img[c] = img[c] * std[c] + mean[c]
    elif img.ndim == 3 and img.shape[2] == 3:  # HWC format
        for c in range(3):
            img[:,:,c] = img[:,:,c] * std[c] + mean[c]
    return np.clip(img * 255, 0, 255).astype(np.uint8)

def enhance_contrast(image: np.ndarray) -> np.ndarray:
    """Enhance contrast using CLAHE."""
    if len(image.shape) == 3:
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l_channel, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl = clahe.apply(l_channel)
        limg = cv2.merge((cl,a,b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    else:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        return clahe.apply(image)

def estimate_skew_angle(image: np.ndarray) -> float:
    """
    Estimate page skew in degrees (cv2.getRotationMatrix2D convention, positive = counter-clockwise).

    Method: minimum-area bounding rectangle of the ink pixels. Ink is separated from the
    background with an inverted Otsu threshold (dark text on light paper). The angle is read
    from the rectangle's corner points rather than minAreaRect's angle field, because that
    field's range changed between OpenCV 4.x and 5.x. Only skews within +/-45 degrees are detectable.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    points = cv2.findNonZero(ink)
    if points is None:
        return 0.0
    corners = cv2.boxPoints(cv2.minAreaRect(points))
    dx, dy = corners[1] - corners[0]
    edge_angle = np.degrees(np.arctan2(dy, dx))
    # Either rectangle edge works once folded into [-45, 45)
    edge_angle = (edge_angle + 45) % 90 - 45
    # Image y points down, so a counter-clockwise tilt shows up as a negative edge angle
    return float(-edge_angle)

def deskew_image(image: np.ndarray) -> np.ndarray:
    """Deskew a document image by rotating it back by its estimated skew."""
    angle = estimate_skew_angle(image)
    if abs(angle) < 0.1:
        return image
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, -angle, 1.0)
    # Fill uncovered corners with white paper; BORDER_REPLICATE smeared edge ink into fake lines
    white = (255,) * image.shape[2] if image.ndim == 3 else 255
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=white)

def preprocess_document(image: np.ndarray) -> np.ndarray:
    """Full preprocessing pipeline for documents."""
    img = deskew_image(image)
    img = enhance_contrast(img)
    return img

def pdf_page_count(pdf_bytes: bytes) -> int:
    import pymupdf
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
        return doc.page_count

def render_pdf_page(pdf_bytes: bytes, page_index: int = 0, dpi: int = 150, max_pixels: int = 40_000_000) -> np.ndarray:
    """
    Render one PDF page to a BGR image (OpenCV channel order). 150 dpi keeps body text legible;
    PyMuPDF's default of 72 dpi shrinks an A4 page to 595x842 px. Oversized pages (posters, maps)
    are rendered at a lower dpi so the image stays under max_pixels.
    """
    import pymupdf
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
        if not 0 <= page_index < doc.page_count:
            raise IndexError(f"page {page_index} out of range (PDF has {doc.page_count} pages)")
        rect = doc[page_index].rect  # in points (1/72 inch)
        pixels_at_dpi = rect.width * rect.height * (dpi / 72) ** 2
        if pixels_at_dpi > max_pixels:
            dpi = int(dpi * (max_pixels / pixels_at_dpi) ** 0.5)
        pix = doc[page_index].get_pixmap(dpi=dpi)
        return cv2.imdecode(np.frombuffer(pix.tobytes("png"), np.uint8), cv2.IMREAD_COLOR)

def decode_image(data: bytes, max_pixels: int = 40_000_000) -> np.ndarray:
    """
    Decode uploaded bytes to a BGR image, refusing images with more than max_pixels pixels.
    The size is read from the file header before decoding, so a small file that declares a huge
    image (a "decompression bomb") is rejected without allocating memory for it.
    """
    import io
    import warnings
    from PIL import Image
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as im:
                width, height = im.size
    except Image.DecompressionBombError as e:
        raise ValueError(f"Image too large: over {max_pixels / 1e6:.0f} megapixels") from e
    except Exception as e:
        raise ValueError("Not a readable image") from e
    if width * height > max_pixels:
        raise ValueError(f"Image too large: {width * height / 1e6:.0f} megapixels (limit {max_pixels / 1e6:.0f})")
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Not a readable image")
    return image
