import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import mean_absolute_error, cohen_kappa_score
from src.data_processing.feature_engineering import data_feature_engineering, split
from src.test_models.test_random_forest_classifier import test_random_forest_classifier

"""model = Sequential([tensorflow.keras.layers.Dense(64, activation='relu', # First layer with 128 neurons
                                                      ), # Here we must specify the input shape
                       Dropout(0.2), # A Dropout layer of 20%
                       Dense(32, activation='relu'), # Second layer with 64 neurons
                       Dense(1, # Output layer with 1 neuron
                              activation='sigmoid')]) # sigmoid function is ideal for binary classification

    # Compile the model
    model.compile(optimizer='adam', # Adam optimizer algorithm
                  loss='binary_crossentropy', # Binary crossentropy for binary classification
                  metrics=['accuracy'])

    # Train the model
    history = model.fit(X_train, y_train, epochs=500, validation_split=0.2)

    # Evaluate the model
    nn_loss, nn_accuracy = model.evaluate(X_test, y_test)
    print(f"Neural Network Accuracy: {nn_accuracy}")"""



def main():
    # Load your data
    df = pd.read_csv('../data/user_feedback.csv')

    df = data_feature_engineering(df)

    X_train, y_train, X_test, y_test = split(df, smote=False)

    test_random_forest_classifier(X_train, y_train, X_test, y_test)



def test_all_models(X_train, X_test, y_train, y_test):

    return





if __name__ == '__main__':
    main()