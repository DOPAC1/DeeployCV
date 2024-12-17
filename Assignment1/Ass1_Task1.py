import cv2
import numpy as np
import matplotlib.pyplot as plt

def show(name, n, m, i, Title):
    plt.subplot(n, m, i)
    plt.imshow(name, cmap='gray')
    plt.title(Title)
    plt.axis('off')

def capture_image():
    cap = cv2.VideoCapture(0)  # Default camera
    ret, frame = cap.read()    # Capture a frame
    cap.release()
    if not ret:
        print("Failed to capture image")
        return None
    return frame

def grey_scale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def threshold_black_white(image, threshold=128):
    _, thresh_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    return thresh_image

def quantize_greyscale(image, levels=16):
    step = 256 // levels
    return (image // step) * step

def sobel_edge_detection(image):
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    sobel = cv2.magnitude(sobel_x, sobel_y)
    return sobel

def canny_edge_detection(image, threshold1=50, threshold2=150):
    return cv2.Canny(image, threshold1, threshold2)

def apply_gaussian_blur(image, kernel_size=(5, 5)):
    return cv2.GaussianBlur(image, kernel_size, 0)

def sharpen_image(image):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])  # Sharpening kernel
    return cv2.filter2D(image, -1, kernel)

def convert_rgb_to_bgr(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

def main():
    # Capture the image
    image = capture_image()
    if image is None:
        return

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray_image = grey_scale(image)
    bw_image = threshold_black_white(gray_image)
    quantized_image = quantize_greyscale(gray_image, levels=16)
    sobel_edges = sobel_edge_detection(gray_image)
    canny_edges = canny_edge_detection(gray_image)
    blurred_image = apply_gaussian_blur(gray_image)
    sharpened_image = sharpen_image(blurred_image)
    bgr_image = convert_rgb_to_bgr(image_rgb)

    plt.figure(figsize=(12, 8))
    show(image_rgb, 2, 4, 1, "Original Image (RGB)")
    show(gray_image, 2, 4, 2, "Grey Scaled")
    show(bw_image, 2, 4, 3, "Black & White Threshold")
    show(quantized_image, 2, 4, 4, "16 Grey Colors")
    show(sobel_edges, 2, 4, 5, "Sobel Edges")
    show(canny_edges, 2, 4, 6, "Canny Edges")
    show(blurred_image, 2, 4, 7, "Gaussian Blurred")
    show(sharpened_image, 2, 4, 8, "Sharpened Image")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
