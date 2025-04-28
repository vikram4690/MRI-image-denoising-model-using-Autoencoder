import gradio as gr
import numpy as np
import tensorflow as tf
from PIL import Image

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="autoencoder_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def denoise_image(input_image):
    input_image = input_image.convert('L')  # Convert to grayscale
    input_image = input_image.resize((64, 64))  # Resize to model size
    image_array = np.array(input_image).astype(np.float32) / 255.0
    input_data = np.expand_dims(image_array, axis=(0, -1))  # Shape: (1, 64, 64, 1)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    denoised_image = output_data[0, :, :, 0]

    denoised_image = (denoised_image * 255).astype(np.uint8)  # Scale back to 0-255

    return Image.fromarray(denoised_image)

# Gradio interface
demo = gr.Interface(
    fn=denoise_image,
    inputs=gr.Image(type="pil", label="Upload Noisy Image"),
    outputs=gr.Image(type="pil", label="Denoised Output"),
    title="🧹 Image Denoiser Autoencoder",
    description="Upload a noisy 64x64 grayscale image. The model will denoise it."
)

if __name__ == "__main__":
    demo.launch()
