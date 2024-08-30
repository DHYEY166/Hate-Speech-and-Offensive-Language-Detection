## Hate Speech and Offensive Language Detection

Welcome to my Data Science Project! This project focuses on detecting hate speech and offensive language using an ensemble machine learning model. The application is built with Streamlit, allowing users to input text and receive a classification indicating whether the text contains hate speech or is non-hate speech.

## Table of Contents
- [App Overview](#app-overview)
- [Installation](#installation)
- [Usage](#usage)
- [Model Details](#model-details)
- [Preprocessing](#preprocessing)
- [Dataset](#dataset)
- [Features](#features)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## App Overview

This app allows users to input a comment for analysis. The application processes the text using an ensemble machine learning model and provides a prediction indicating whether the text is classified as hate speech or non-hate speech.

## Installation

To run this application locally, follow these steps:

1. Clone the repository:
   
   git clone https://github.com/DHYEY166/Hate-Speech-and-Offensive-Language-Detection.git
   
   cd Hate-Speech-and-Offensive-Language-Detection

3. Create a virtual environment (optional but recommended):

   python -m venv venv
   
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`

5. Install the required packages:

   pip install -r requirements.txt

6. Run the Streamlit application:

   streamlit run app.py

## Usage

- **Input a Comment**: Enter the text you want to analyze in the provided text area.
- **Analyze**: Click the "Analyze" button to predict whether the comment is hate speech or non-hate speech.
- **View Result**: The result will be displayed on the page, indicating whether the text is classified as "Hate Speech" or "Non-Hate Speech."

You can also access the application directly via the following link:

[Streamlit Application](https://hate-speech-and-offensive-language-detection-6cahrrrxj3eqeocx7.streamlit.app)

## Model Details

The application utilizes an ensemble model consisting of Logistic Regression, Multinomial Naive Bayes, and Decision Tree classifiers. The model was trained on two Kaggle datasets focused on hate speech and offensive language detection.

## Preprocessing

- **Text Cleaning**: The text is cleaned by removing URLs, punctuation, numbers, and stopwords, and then stemmed using the Snowball Stemmer.
- **Vectorization**: The text data is vectorized using TfidfVectorizer, limiting to 5000 features with n-gram range (1,2).

## Dataset

The model was trained using the following datasets(available on Kaggle):

- Hate Speech and Offensive Language Dataset [Hate Speech and Offensive Language Dataset](https://www.kaggle.com/datasets/mrmorj/hate-speech-and-offensive-language-dataset)
- Twitter Sentiment Analysis: Hatred Speech [Twitter Sentiment Analysis: Hatred Speech](https://www.kaggle.com/datasets/arkhoshghalb/twitter-sentiment-analysis-hatred-speech)

## Features

- **Text Input**: Users can input any text they wish to analyze.
- **Model Prediction**: The model classifies the input text as either "Hate Speech" or "Non-Hate Speech."
- **Visualization**: The result is displayed directly on the application page for easy interpretation.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the [LICENSE](https://github.com/DHYEY166/BREAST_CANCER_SEMANTIC_SEGMENTATION/blob/main/LICENSE) file for more details.

## Contact

- **Author**: Dhyey Desai
- **Email**: dhyeydes@usc.edu
- **GitHub**: https://github.com/DHYEY166
- **LinkedIn**: https://www.linkedin.com/in/dhyey-desai-80659a216 

Feel free to reach out if you have any questions or suggestions.
