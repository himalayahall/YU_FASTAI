import boto3

# s3 = boto3.client('s3')
# s3.download_file('jhfastai', 'bears.pkl', '/Users/jawaidhakim/Downloads/bears3.pkl')

from fastai.vision.all import *
from fastbook import *
import streamlit as st


class Predict:

    def __init__(self, model_path, model_name):

        # Model (pkl file) path
        self.model_path = model_path

        # Model (pfk file) name
        self.mode_name = model_name

        # Load saved model
        self.path = Path(self.model_path)
        self.learn_inference = load_learner(self.path / self.mode_name)

        self.img = None

    # Image selected callback
    def on_image_selected(self) -> None:

        # Load selected image
        if 'uploaded_file' in st.session_state:
            uploaded_file = st.session_state.uploaded_file
            if uploaded_file is not None:
                self.img = PILImage.create(uploaded_file)

        if self.img is not None:
            self.show_image()
            st.button('Classify', on_click=self.on_classify_clicked)
            st.write(f'Click button to classify')

    # Show selected image
    def show_image(self):
        if self.img is not None:
            st.image(self.img.to_thumb(300, 300), caption='Uploaded Image')

    # Callback when classification button is clicked
    def on_classify_clicked(self) -> None:
        if self.img is not None:
            self.show_image()
            pred, pred_idx, probs = self.learn_inference.predict(self.img)
            st.write(f'## Prediction: {pred}; Probability: {probs[pred_idx]:.04f}')
        else:
            st.write("image is null")


import sys

if __name__ == '__main__':
    # col = st.columns(1)
    # with col:

    if len(sys.argv) < 3:
        st.error("Missing model file path and name")
        st.snow()
        st.stop()

    # Instantiate predictor
    #predictor = Predict('/Users/jawaidhakim/Downloads', 'bears.pkl')
    predictor = Predict(sys.argv[1], sys.argv[2])

    # Write header
    st.header('Bear Classifier')

    # Create file uploader with callback, name of selected file sent in session_state by key
    st.file_uploader("Upload Files", type=['png', 'jpeg', 'jpg'], on_change=predictor.on_image_selected,
                     key='uploaded_file')