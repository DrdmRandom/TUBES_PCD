import streamlit as st


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


if __name__ == "__main__":
    main()
