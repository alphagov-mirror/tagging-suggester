from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np
import src.utils.nlp as nlp
from src.data.representative_content import *
from src.utils import directories as dirs
import os as apex_node_os


class ApexNodePredictor:
    def train(self, tree):
        """
        Trains models and generates files so instances of this class
        can make predictions on which apex node(s) a content best belongs to
        """
        self.tree = tree
        texts, y = self.training_data()
        X = self.fit_vectorizer(texts)
        self.train_model(X, y)

    def training_data(self):
        """
        Data needed for training
        :returns: List. 1. List of all texts, 2. List of classes for texts
        """
        texts = []
        y = []
        for apex_node in self.tree.apex_nodes() :
            texts_for_apex_node = RepresentativeContent.representative_content_for_taxon(apex_node)
            texts += texts_for_apex_node
            y += [apex_node.content_id] * len(texts_for_apex_node)
        return [texts, y]

    def fit_vectorizer(self, texts):
        """
        Fits TF-IDF vectorizer and saves it to file
        :returns: X matrix for training
        """
        vectorizer = TfidfVectorizer(tokenizer=nlp.tokenize, analyzer='word', stop_words='english', max_features=1000 )
        X = vectorizer.fit_transform(texts).toarray()
        dirs.save_pickle_file(vectorizer, self.vectorizer_file_path())
        return X

    def train_model(self, X, y):
        """
        Trains model and saves it to file
        :param X: matrix of parameters training
        :param y: List, true classes for training
        """
        model = LogisticRegression(solver='liblinear', multi_class='ovr', max_iter=200).fit(X, y)
        model.fit(X, y)
        dirs.save_pickle_file(model, self.model_file_path())

    def model_file_path(self):
        return apex_node_os.path.join(dirs.processed_data_dir(), "models", "apex_node", "apex_node_model.pkl")

    def vectorizer_file_path(self):
        return apex_node_os.path.join(dirs.processed_data_dir(), "models", "apex_node", "apex_node_vectorizer.pkl")

    def threshold(self):
        """
        Threshold of probability above which we consider a prediction
        to be good enough. Tested in order to find the best F1 score
        """
        return 0.225

    def predict(self, tree, text_to_predict, request_record):
        """
        Predicts which apex nodes content should be tagged to
        :param text_to_predict: String, text to predict
        :return: List, names of apex taxons the content should be tagged to, unsorted.
        """
        vectorizer = dirs.open_pickle_file(self.vectorizer_file_path())
        model = dirs.open_pickle_file(self.model_file_path())
        class_probabilities = model.predict_proba(vectorizer.transform([text_to_predict]))[0]
        all_results = []
        results = {}
        for i, probability in enumerate(class_probabilities):
            predicted_node_content_id = model.classes_[i]
            is_above_threshold = probability > self.threshold()
            all_results.append({ 'node_content_id': predicted_node_content_id, 'probability': probability, 'above_threshold': is_above_threshold })
            if is_above_threshold:
                results[tree.find(predicted_node_content_id)] = predicted_node_content_id
        request_record.apex_node_predictor_probabilities = str(all_results)
        # Order results by probability (highest first)
        return [sorted(results, key=results.get, reverse=True), request_record]
