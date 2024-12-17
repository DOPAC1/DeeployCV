import cv2
import numpy as np
import matplotlib.pyplot as plt


# Function to show images in a grid
def show(name, n, m, i, Title):
    plt.subplot(n, m, i)
    plt.imshow(name, cmap='gray')
    plt.title(Title)
    plt.axis('off')


# Stronger high-pass filter (sharpening using a kernel)
def high_pass_filter(image):
    # Stronger sharpening kernel
    kernel = np.array([[-1, -1, -1],
                       [-1, 8, -1],
                       [-1, -1, -1]])
    return cv2.filter2D(image, -1, kernel)


# Low-pass filter (Gaussian blur)
def low_pass_filter(image, kernel_size=(15, 15)):
    return cv2.GaussianBlur(image, kernel_size, 0)


# Combine two images (add weights) after resizing
def combine_images(image1, image2, alpha=0.5, beta=0.5):
    # Resize image2 to match the shape of image1
    image2_resized = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
    return cv2.addWeighted(image1, alpha, image2_resized, beta, 0)


# Main Execution
def main():
    # Load two local images (grayscale)
    image1_path = 'Pikachu Png Transparent Image - Pikachu Png, Png Download , Transparent Png Image - PNGitem.jpeg'  # Replace with the path to your first image
    image2_path = 'Chill_guy_original_artwork.jpg'  # Replace with the path to your second image

    # Read images as grayscale
    image1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)

    if image1 is None or image2 is None:
        print("Error: Could not load images. Check file paths.")
        return

    # Apply filters
    high_pass_image = high_pass_filter(image1)
    low_pass_image = low_pass_filter(image2)
    combined_image = combine_images(high_pass_image, low_pass_image)

    # Plot all five images
    plt.figure(figsize=(15, 10))

    show(image1, 2, 3, 1, "Original Image 1")
    show(image2, 2, 3, 2, "Original Image 2")
    show(high_pass_image, 2, 3, 3, "High-Pass Filtered Image")
    show(low_pass_image, 2, 3, 4, "Low-Pass Filtered Image")
    show(combined_image, 2, 3, 5, "Combined Image")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
