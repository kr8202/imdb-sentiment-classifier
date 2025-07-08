IMDb Movie Reviews Sentiment Analysis

This project demonstrates the application of Natural Language Processing (NLP) and supervised machine learning techniques to classify movie reviews from IMDb as positive or negative. The goal is to build reliable sentiment classifiers using text data and evaluate their performance.

📌 Project Objective

To develop, train, and evaluate machine learning models capable of predicting sentiment polarity (positive or negative) based on IMDb movie review texts. This project aims to showcase how textual data can be preprocessed and modeled using classical ML techniques for real-world applications.

🧰 Technologies & Tools Used
Language: Python

Libraries & Frameworks:

pandas, numpy – Data manipulation
nltk – Text preprocessing (tokenization, stopword removal, stemming)
scikit-learn – Feature extraction, model training, and evaluation
matplotlib, seaborn – Data visualization

⚙️ Methodology
Data Preprocessing
Removed noise and HTML tags from text
Tokenized sentences
Removed stopwords
Applied stemming using NLTK
Feature Extraction
Used TF-IDF Vectorization to convert text into a numerical format
Also experimented with Word2Vec embeddings (optional)
Model Training

Trained multiple classification models:

- Logistic Regression
- Naive Bayes
- Random Forest
- Model Evaluation

Used accuracy, precision, recall, and F1-score to evaluate performance
Visualized results with a confusion matrix and metric comparison charts

📊 Results
Achieved high sentiment classification accuracy across models
Logistic Regression and Naive Bayes provided consistent results with clean TF-IDF features

Folder Structure:
├── Imdb.ipynb                   # Main Jupyter Notebook containing the entire workflow
├── README.md                    # Project documentation and overview
├── requirements.txt             # Python libraries and dependencies (optional)
└── data/                        # IMDb dataset directory (can be downloaded from Kaggle)

📥 Dataset Source:
IMDb Review Dataset available on Kaggle
🔗 https://www.kaggle.com/datasets/c134koyenaroy/imdb-review-dataset
Word2Vec added semantic value but required more computational resources

🔍 Key Learnings
Preprocessing significantly impacts model performance in NLP tasks
Classical ML models can perform well on sentiment tasks with clean data and proper feature extraction
TF-IDF is effective and interpretable for text classification the dataset: https://www.kaggle.com/datasets/c134koyenaroy/imdb-review-dataset 
