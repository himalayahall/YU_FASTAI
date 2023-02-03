from fastai.vision.all import *
from fastbook import *
import streamlit as st
import boto3
import io
import sys
import os
import random

prob_msgs = {
                'very low': ["Not confident about this", "To be taken with a grain of salt", 'I have a bad feeling about this'],
                'low': ['Possibly', 'Not feeling very confident'],
                'high': ['Quite likely', 'Quite possibly', 'I think it is'],
                'very high': ['Most likely', 'Quite sure it is', 'It is']
             }

# Load pickled model from S3
@st.cache(suppress_st_warning=True, show_spinner=False, hash_funcs={Learner: lambda _: None})
def load_model_from_s3(s3_bucket, path_to_model):
    # Connect to s3 bucket.
    # AWS credentials must be set up beforehand by running 'aws configure' or
    # by creating .streamlit/secrets.toml with the following:
    #     AWS_ACCESS_KEY_ID=xxxx
    #     AWS_SECRET_ACCESS_KEY=yyyy
    #     AWS_DEFAULT_REGION=us-east-1
    #
    # Make sure this file IS NOT committed to github!
    #
    # Go to the Streamlit app dashboard and in the app 's dropdown menu, click on Edit Secrets.
    # Copy the content of secrets.toml
    s3client = boto3.client('s3')

    # load model from s3 bucket
    response = s3client.get_object(Bucket=s3_bucket, Key=path_to_model)
    body = response['Body'].read()

    # IMPORTANT: MUST convert raw bytes into in-memory bytes buffer for random access by torch
    byte_stream = io.BytesIO(body)
    st.success("Loaded model successfully")

    # Learner can be created from bytes!
    return load_learner(byte_stream)


learn_inference = None


class Predict:
    def __init__(self, s3_bucket, s3_path_to_model):

        global learn_inference
        # Load saved model
        if learn_inference is None:
            with st.spinner("Loading model " + s3_path_to_model + " from S3 bucket " + s3_bucket + "..."):
                learn_inference = load_model_from_s3(s3_bucket, s3_path_to_model)

        self.s3_bucket = s3_bucket
        self.s3_path_to_model = s3_path_to_model
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
            pred, pred_idx, probs = learn_inference.predict(self.img)
            prob = float(probs[pred_idx])

            # confidence appropriate msg
            if prob >= 0.95:
                msgs = prob_msgs['very high']
            elif prob >= 80.0:
                msgs = prob_msgs['high']
            elif prob >= 65.0:
                msgs = prob_msgs['low']
            else:
                msgs = prob_msgs['very low']
            msg_idx = random.randint(0, len(msgs))
            msg = msgs[msg_idx]

            st.write(f'## {msg}: {pred} (Prob: {prob:.04f})')

            if prob >= 0.95 and random.randint(1, 10) > 8: # show balloons 20% of time
                st.balloons()

        else:
            st.write("image is null")


def s3_bucket_and_model():
    s3_bucket_name = None
    s3_model_path = None

    # Pick up config first from command line followed bhy env
    if len(sys.argv) >= 3:
        s3_bucket_name = sys.argv[1]
        s3_model_path = sys.argv[2]
    else:
        s3_bucket_name = os.environ.get('s3_bucket_name')
        s3_model_path = os.environ.get('s3_model_path')

    if s3_bucket_name is None or s3_model_path is None:
        if s3_bucket_name is None:
            st.error("Missing S3 bucket name")
        if s3_model_path is None:
            st.error("Missing model path")
        st.snow()
        st.stop()

    return s3_bucket_name, s3_model_path


if __name__ == '__main__':
    s3_bucket_name, s3_model_path = s3_bucket_and_model()

    # Instantiate predictor
    # predictor = Predict('/Users/jawaidhakim/Downloads', 'bears.pkl')
    predictor = Predict(s3_bucket_name, s3_model_path)

    # Write header
    st.header('Bear Classifier')

    # Create file uploader with callback, name of selected file sent in session_state by key
    st.file_uploader("Upload Files", type=['png', 'jpeg', 'jpg'], on_change=predictor.on_image_selected,
                     key='uploaded_file')
