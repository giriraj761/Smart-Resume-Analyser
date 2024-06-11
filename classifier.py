import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
import pickle

class ResumeClassifier:
    def __init__(self):
        # Initialize the vectorizer and the classifier
        self.vectorizer = CountVectorizer()
        self.classifier = KNeighborsClassifier(n_neighbors=3)

    def train(self, X_train, y_train):
        """
        Train the KNN classifier with the training data
        """
        # Convert the text data to feature vectors
        X_train_vectors = self.vectorizer.fit_transform(X_train)
        
        # Fit the KNN classifier
        self.classifier.fit(X_train_vectors, y_train)

    def predict(self, X_test):
        """
        Predict the class labels for the input data
        """
        # Convert the text data to feature vectors
        X_test_vectors = self.vectorizer.transform(X_test)
        
        # Predict using the trained classifier
        return self.classifier.predict(X_test_vectors)

    def save_model(self, vectorizer_path='vectorizer.pkl', classifier_path='classifier.pkl'):
        """
        Save the vectorizer and the classifier to disk
        """
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
        
        with open(classifier_path, 'wb') as f:
            pickle.dump(self.classifier, f)

    def load_model(self, vectorizer_path='vectorizer.pkl', classifier_path='classifier.pkl'):
        """
        Load the vectorizer and the classifier from disk
        """
        with open(vectorizer_path, 'rb') as f:
            self.vectorizer = pickle.load(f)
        
        with open(classifier_path, 'rb') as f:
            self.classifier = pickle.load(f)

if __name__ == "__main__":
    X_train = ["Experienced in machine learning and data science",
               "Web developer with knowledge of JavaScript and React",
               "Software engineer specializing in backend development"]
    y_train = ["Data Science", "Web Development", "Software Engineering"]

    classifier = ResumeClassifier()
    classifier.train(X_train, y_train)
    
    X_test = ["Proficient in Python and machine learning algorithms",
              "Frontend developer with skills in HTML, CSS, and JavaScript"]
    predictions = classifier.predict(X_test)
    
    for resume, prediction in zip(X_test, predictions):
        print(f"Resume: {resume}\nPredicted Category: {prediction}\n")
    
    classifier.save_model()
