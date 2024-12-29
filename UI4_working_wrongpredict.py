import torch
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
from PIL import Image
import streamlit as st

# List of Classes
classes = [
    'Bangus', 'barrel_jellyfish', 'Corals', 'compass_jellyfish', 'Climbing Perch', 'Clams',
    'Dolphin', 'blue_jellyfish', 'Crabs', 'Catfish', 'Black Spotted Barb', 'Big Head Carp',
    'Fourfinger Threadfin', 'Goby', 'Gold Fish', 'Indo-Pacific Tarpon', 'Indian Carp',
    'Green Spotted Puffer', 'Freshwater Eel', 'Grass Carp', 'Gourami', 'Glass Perchlet',
    'Jaguar Gapote', 'Mosquito Fish', 'mauve_stinger_jellyfish', 'Janitor Fish', 'Moon_jellyfish',
    'lions_mane_jellyfish', 'Knifefish', 'Mudfish', 'Lobster', 'Long-Snouted Pipefish',
    'Mullet', 'Nudibranchs', 'Puffers', 'Otter', 'Scat Fish', 'Perch', 'Pangasius', 'Penguin',
    'Sea Rays', 'Octopus', 'Tenpounder', 'Silver Barb', 'Snakehead', 'Silver Perch',
    'Silver Carp', 'Tilapia'
]

# Define the function to load the model
@st.cache_resource
def load_model():
    # Define the model architecture
    num_classes = len(classes)  # Use the length of the classes list
    model = models.resnet50(weights=None)  # Initialize a ResNet50 model without pretrained weights
    model.fc = nn.Linear(model.fc.in_features, num_classes)  # Modify the final layer for num_classes

    # Load the state dictionary
    state_dict = torch.load("best_model.pth", map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)

    # Set the model to evaluation mode
    model.eval()
    return model

# Define a function for preprocessing the image
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

# Define a function to make predictions
def predict(image, model):
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        return classes[predicted.item()]  # Map index to class name

# Define a function to make predictions with confidence
def predict_with_confidence(image, model):
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        top_prob, top_class = probabilities.topk(1, dim=1)
        return classes[top_class.item()], top_prob.item()  # Map index to class name and return confidence

def main():
    st.markdown(
        """
        <style>
        .block-container {
            padding: 20px;
            background-color: #D3D3D3;
            border-radius: 10px;
            text-align: center;
            margin: 20px;
        }
        .button {
            margin-top: 10px;
            background-color: #0000FF;
            color: white;
            font-size: 16px;
            border-radius: 5px;
            padding: 10px 20px;
            border: none;
            cursor: pointer;
        }

        </style>

        <div class="block-container">
            <h1>Fish Species Detector</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader("Upload an image of a fish", type=["jpg", "jpeg", "png"])
    model = load_model()

    if uploaded_file is not None:
        # Load the image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Detect Fish Species"):
            with st.spinner("Detecting..."):
                # Preprocess the image
                input_tensor = preprocess_image(image)

                # Make prediction with confidence
                species_name, confidence = predict_with_confidence(input_tensor, model)

                # Display result
                st.success(f"This fish is classified as: {species_name} (Confidence: {confidence:.2f})")

if __name__ == "__main__":
    main()
