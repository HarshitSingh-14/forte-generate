# Custom Style Response Generator

This application generates responses in different styles (Email, Normal, Report, Feedback) based on user input and context. It uses a Retrieval-Augmented Generation (RAG) approach with a word limit to ensure responses are concise and style-aware.
This is a follow-up to the previous project named `style-admin` and uses the styles created using the `style-admin` by processing documents in s3 bucket.
 This app is hosted at [streamlit](https://llm-style-app.streamlit.app).

## Features
- Choose from Email, Normal, Report, or Feedback styles.
- Adjust response creativity with Temperature Control.
- Set how much external knowledge the model can use.
- Limit the word count for generated responses.
- Generate context-based responses using provided input.


## Installation

To run the app locally, follow these steps:

1. Clone the repository:
    ```sh
    git clone https://github.com/ayushpratap113/style-generate.git
    cd style-generate
    ```

2. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

3. Run the Streamlit app:
    ```sh
    streamlit run src/app.py
    ```

## Environment Variables

Ensure you have a [.env](http://_vscodecontentref_/1) file with the following variables:

```env
BUCKET_NAME="your-bucket-name"
AWS_REGION="your-aws-region"
AWS_ACCESS_KEY_ID="your-access-key-id"
AWS_SECRET_ACCESS_KEY="your-secret-access-key"
