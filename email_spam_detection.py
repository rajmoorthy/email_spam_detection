import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load and preprocess the dataset
@st.cache_data
def load_data():
    data = pd.read_csv('C:/Users/91936/Desktop/raj/spam.csv', encoding='latin1')
    data = data[['v1', 'v2']]
    data.columns = ['Label', 'Email']
    data['Label'] = data['Label'].map({'spam': 1, 'ham': 0})
    return data

# Train the Naive Bayes model and get evaluation metrics
@st.cache_data
def train_model(data):
    X = data['Email']
    y = data['Label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    vectorizer = CountVectorizer(stop_words='english')
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)
    
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    return model, vectorizer, accuracy, conf_matrix, report

# Streamlit app interface
st.title("Email Spam Detection with Naive Bayes")
st.write("This app uses a Naive Bayes model to classify emails as 'Spam' or 'Not Spam'.")

# Load data and train model
data = load_data()
model, vectorizer, accuracy, conf_matrix, report = train_model(data)

# Display model accuracy
st.write(f"Model Accuracy: {accuracy * 100:.2f}%")

# Confusion Matrix Visualization
st.write("Confusion Matrix:")
fig, ax = plt.subplots()
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix for Naive Bayes Model')
st.pyplot(fig)

# Email classification
st.write("## Test the Model with a New Email")
user_input = st.text_area("Enter email content here")

# Prediction function for new email
if st.button("Predict"):
    if user_input:
        email_vec = vectorizer.transform([user_input])
        prediction = model.predict(email_vec)
        result = "Spam" if prediction[0] == 1 else "Not Spam"
        st.write(f"The email is classified as: **{result}**")
    else:
        st.write("Please enter some text to classify.")

# Display classification report
st.write("### Classification Report:")
st.json(report)
