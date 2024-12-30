import os
import torch
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
from PIL import Image
import streamlit as st

# List of Classes
train_dir = "./Dataset/train"  # Use relative path for Linux/Podman environment
if not os.path.exists(train_dir):
    raise ValueError(f"Training directory not found: {train_dir}")

# Ensure consistent class sorting by applying `sorted()`
class_names = sorted(os.listdir(train_dir))  # Get a sorted list of class names

# Debugging: Verify class names and their order
print("Class Names:", class_names)
print("Number of Classes:", len(class_names))


# Create the model
def create_model(num_classes):
    model = models.resnet50(weights=None)  # Initialize a ResNet50 model without pretrained weights
    model.fc = nn.Linear(model.fc.in_features, num_classes)  # Modify the final layer for num_classes
    return model


# Load the model
def load_model():
    num_classes = len(class_names)  # Use the number of classes from the dataset
    model = create_model(num_classes)

    # Load the checkpoint
    state_dict = torch.load("best_model.pth", map_location=torch.device("cpu"))

    # Debugging: Check keys in state_dict
    print("Keys in state_dict:", state_dict.keys())

    # Ensure strict=True to catch mismatches (optional, for debugging)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        print(f"Error loading model state_dict: {e}")
        # If necessary, remove mismatched keys
        if "fc.weight" in state_dict:
            print("Removing fc.weight from state_dict")
            del state_dict["fc.weight"]
        if "fc.bias" in state_dict:
            print("Removing fc.bias from state_dict")
            del state_dict["fc.bias"]
        model.load_state_dict(state_dict, strict=False)

    model.eval()  # Set model to evaluation mode
    return model


# Preprocess the image
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Ensure same stats
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension


# Predict with top-3 results
def predict_top3(input_tensor, model, class_names):
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

    # Debugging: Print probabilities and class names
    print("Probabilities:", probabilities)
    print("Number of Classes in Model Output:", probabilities.size(0))
    print("Class Names:", class_names)

    # Ensure k does not exceed the number of classes
    k = min(3, probabilities.size(0))
    top3_prob, top3_indices = torch.topk(probabilities, k)

    # Map indices to class names
    top3_classes = [class_names[idx] for idx in top3_indices]
    return list(zip(top3_classes, top3_prob.tolist()))


def main():
    st.title("Fish Species Detector")
    st.write("Upload an image of a fish to get its predicted species and confidence levels.")
    st.markdown(
        """
        <style>
        [title~="Deploy"] {
            display: none;
        }
        .footer {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            background-color: #262730;
            color: white;
            text-align: center;
            padding: 10px 0;
            font-size: 14px;
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

        # Detect species
        if st.button("Detect Fish Species"):
            with st.spinner("Classifying..."):
                input_tensor = preprocess_image(image)  # Preprocess image
                predictions = predict_top3(input_tensor, model, class_names)  # Get top-3 predictions

                # Display predictions
                st.subheader("Top Predictions:")
                for species, confidence in predictions:
                    st.write(f"- **{species}**: {confidence:.2%}")

    # Footer
    st.markdown(
        """
        <div class="footer">
            <strong>Created By</strong><br>
            Anaz & Dawwi<br>
            Telkom University <br> 
            International Informatics Class Student 
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
