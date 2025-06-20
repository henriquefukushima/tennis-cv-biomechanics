'''
import numpy as np

def hsv_cv2hsv(hsv = [0, 0, 0]):

    """
    Converte uma lista HSV (Hue, Saturation, Value) para o formato HSV do OpenCV.

    Args:
        hsv (list): Lista contendo os valores de Hue, Saturation e Value.

    Returns:
        numpy.ndarray: Imagem em formato HSV do OpenCV.
    """

    # Hue entre 0 e 179
    hsv[0] = int(hsv[0] * 179 / 360)

    # Saturation e Value entre 0 e 255
    hsv[1] = int(hsv[1] * 255 / 100)
    hsv[2] = int(hsv[2] * 255 / 100)

    return np.array(hsv, dtype=np.uint8)
'''
import numpy as np

def hsv_cv2hsv(h, s, v):
    """
    Convert HSV given in *degrees* (H 0-360) and *percent* (S, V 0-100)
    to OpenCV's 8-bit HSV (H 0-179, S,V 0-255).

    Returns
    -------
    np.ndarray  shape (3,)  dtype uint8
    """
    return np.array([
        int(h / 2),               # Hue: 0-360 → 0-179  (OpenCV uses half-degrees)
        int(s / 100 * 255),       # Sat: 0-100% → 0-255
        int(v / 100 * 255)        # Val: 0-100% → 0-255
    ], dtype=np.uint8)
