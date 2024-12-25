from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf
from tf_explain.core.grad_cam import GradCAM

# Indices corresponding to the class to be visualized (817 = "sports car" in ImageNet)
indices = [817, 817]
# Paths to the car images
IMAGE_PATHS = ["m1.jpg", "m2.jpeg"]  # Replace with your actual image file paths

# Load pre-trained VGG16 model
model = tf.keras.applications.VGG16(weights="imagenet", include_top=True)

# Loop through the image paths
for i in range(len(IMAGE_PATHS)):
    each_path = IMAGE_PATHS[i]
    index = indices[i]

    # Load and preprocess the image
    img = load_img(each_path, target_size=(224, 224))
    img = img_to_array(img)
    img = tf.keras.applications.vgg16.preprocess_input(img)  # Preprocess input for VGG16
    data = ([img], None)

    # Define the name to save the Grad-CAM visualization
    name = each_path.split("/")[-1].split(".")[0]  # Extract name without file extension

    # Generate Grad-CAM visualization
    explainer = GradCAM()
    grid = explainer.explain(
        validation_data=data,           # Data to visualize
        model=model,                    # Pre-trained model
        layer_name='block5_conv3',      # Layer name to visualize
        class_index=index               # Class index for Grad-CAM
    )

    # Save the Grad-CAM visualization
    explainer.save(grid, '.', name + '_grad_cam.png')
    print(f"Grad-CAM saved for {each_path} as {name}_grad_cam.png")
