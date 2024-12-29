import os
import torch
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
from PIL import Image
import streamlit as st

# List of Classes
train_dir = r"C:\Users\dawwi\OneDrive\Desktop\ui_pcd\Dataset\train"  # Path to the training dataset
class_names = sorted(os.listdir(train_dir))  # Get a sorted list of class names

# Print the class names
print("Class Names:", class_names)
print("Number of Classes:", len(class_names))

# Create the model
def create_model(num_classes):
    model = models.resnet50(weights=None)  # Initialize a ResNet50 model without pretrained weights
    model.fc = nn.Linear(model.fc.in_features, num_classes)  # Modify the final layer for num_classes
    return model

# Load the model
@st.cache_resource
def load_model():
    num_classes = len(class_names)
    model = create_model(num_classes)
    state_dict = torch.load("best_model.pth", map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    model.eval()  # Set to evaluation mode
    return model

# Preprocess the image
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

# Predict with top-3 results
def predict_top3(image, model, class_names):
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        top3_prob, top3_indices = torch.topk(probabilities, 3)
    return [(class_names[idx], prob.item()) for idx, prob in zip(top3_indices, top3_prob)]

# Main Streamlit app
def main():
    st.title("Fish Species Detector")
    st.write("Upload an image of a fish to get its predicted species and confidence levels.")
    st.markdown(
        """
        <style>
        [title~="Deploy"] {
            display: none;
        }
        .center-button {
            display: flex;
            justify-content: center;
            margin-top: 20px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    model = load_model()

    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Center the "Detect Fish Species" button
        st.markdown('<div class="center-button">', unsafe_allow_html=True)
        if st.button("Detect Fish Species"):
            with st.spinner("Classifying..."):
                input_tensor = preprocess_image(image)  # Preprocess image
                predictions = predict_top3(input_tensor, model, class_names)  # Get top-3 predictions

                # Display predictions
                st.subheader("Top Predictions:")
                for species, confidence in predictions:
                    st.write(f"- **{species}**: {confidence:.2%}")
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
