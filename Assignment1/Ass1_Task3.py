from PIL import Image
import numpy as np

# Function to determine the dominant color of a region
def get_dominant_color(image_np, threshold=100):
    red_channel = image_np[:, :, 0]
    green_channel = image_np[:, :, 1]
    blue_channel = image_np[:, :, 2]

    # Define red and white pixel ranges
    red_pixels = (red_channel > threshold) & (green_channel < threshold) & (blue_channel < threshold)
    white_pixels = (red_channel > threshold) & (green_channel > threshold) & (blue_channel > threshold)

    # Count red and white pixels
    red_count = np.sum(red_pixels)
    white_count = np.sum(white_pixels)

    if red_count > white_count:
        return 'red'
    else:
        return 'white'

# Main function to determine the flag type
def is_indonesia_or_poland_flag(image_path):
    # Open the image and convert it to RGB
    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    # Split the image into top and bottom halves
    top_half = image.crop((0, 0, width, height // 2))
    bottom_half = image.crop((0, height // 2, width, height))

    # Convert to NumPy arrays for color analysis
    top_half_np = np.array(top_half)
    bottom_half_np = np.array(bottom_half)

    # Detect the dominant color of the top and bottom halves
    top_half_color = get_dominant_color(top_half_np)
    bottom_half_color = get_dominant_color(bottom_half_np)

    # Determine the flag based on color dominance
    if top_half_color == 'red' and bottom_half_color == 'white':
        return "This is the flag of Indonesia."
    elif top_half_color == 'white' and bottom_half_color == 'red':
        return "This is the flag of Poland."

# Main execution
if __name__ == "__main__":
    image_path = "indo.jpg"  
    result = is_indonesia_or_poland_flag(image_path)
    print(result)
