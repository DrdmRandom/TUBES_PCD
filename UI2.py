import streamlit as st
from PIL import Image
import io
import numpy as np

def dummy_model(image_array):
    # This is a placeholder for your actual model
    # Replace this function with your model's prediction code
    return "Example Fish Species"


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
        .file-input {
            width: 80%;
            height: 40px;
            border: 2px solid black;
            border-radius: 5px;
            padding: 5px;
            margin: 20px auto;
            display: block;
        }
        </style>

        <div class="block-container">
            <h1>Fish Species Detector</h1>
            <input type="file" class="file-input" />
            <div>
                <button class="button">Upload</button>
                <button class="button">Use Camera</button>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader("Upload an image of a fish", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Load the image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Prepare image for model
        image_array = np.array(image)

        if st.button("Detect Fish Species"):
            with st.spinner("Detecting..."):
                # Dummy model function
                result = dummy_model(image_array)
                st.success(f"This fish is classified as: {result}")


if __name__ == "__main__":
    main()
