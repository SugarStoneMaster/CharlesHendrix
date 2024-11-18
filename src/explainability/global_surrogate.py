import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import cross_val_predict
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

from src.data_processing.feature_engineering import data_feature_engineering


def global_surrogate_regressor():
    df = pd.read_csv('../../data/user_feedback.csv')
    df = data_feature_engineering(df, mode='fit')
    y = df['UserInput']
    X = df.drop(columns=['UserInput'])


    # Generate out-of-fold predictions for the black-box model
    black_box_model = RandomForestRegressor(n_estimators=160, criterion="friedman_mse", max_depth=28,
                                            min_samples_split=2, min_samples_leaf=1, max_features="log2",
                                            bootstrap=False, random_state=42)
    y_pred_oof = cross_val_predict(black_box_model, X, y, cv=5)

    # Train the black-box model on the entire dataset
    black_box_model.fit(X, y)

    # Train the surrogate model on the full dataset and out-of-fold predictions
    surrogate_model = DecisionTreeRegressor(max_depth=3, random_state=42)
    surrogate_model.fit(X, y_pred_oof)

    # Evaluate the surrogate model’s fidelity on the out-of-fold predictions
    y_surrogate_pred = surrogate_model.predict(X)
    r2 = r2_score(y_pred_oof, y_surrogate_pred)
    print(f"R^2 of surrogate model on out-of-fold predictions: {r2:.2f}")

    # Interpret surrogate model with SHAP
    explainer = shap.TreeExplainer(surrogate_model)
    shap_values = explainer.shap_values(X)
    shap.summary_plot(shap_values, X, plot_type="dot")


def global_surrogate_classifier():
    df = pd.read_csv('../../data/user_feedback.csv')
    df = data_feature_engineering(df, mode='fit')
    y = df['UserInput']
    X = df.drop(columns=['UserInput'])

    # Generate out-of-fold predictions for the black-box model
    black_box_model = RandomForestClassifier(n_estimators=85, criterion="entropy", max_depth=10,
                                             min_samples_split=3, min_samples_leaf=2, max_features="sqrt",
                                             bootstrap=False, random_state=42)
    y_pred_oof = cross_val_predict(black_box_model, X, y, cv=5)  # Using 5-fold cross-validation

    # Train the black-box model on the entire dataset (for deployment purposes)
    black_box_model.fit(X, y)

    # Train the surrogate model on the full dataset and out-of-fold predictions
    surrogate_model = DecisionTreeClassifier(max_depth=3, random_state=42)
    surrogate_model.fit(X, y_pred_oof)

    # Evaluate the surrogate model’s fidelity on the out-of-fold predictions using accuracy
    y_surrogate_pred = surrogate_model.predict(X)
    accuracy = accuracy_score(y_pred_oof, y_surrogate_pred)
    print(f"Accuracy of surrogate model on out-of-fold predictions: {accuracy:.2f}")

    # Interpret surrogate model with SHAP
    explainer = shap.TreeExplainer(surrogate_model)
    shap_values = explainer.shap_values(X)

    # If it's a binary classifier, SHAP might return shap_values as a single array. Check the shape:
    if isinstance(shap_values, list) and len(shap_values) > 1:
        # Multiclass: select a single class, or average across classes
        shap_values_to_plot = shap_values[1]  # E.g., class 1
    else:
        # Binary classification, single set of SHAP values
        shap_values_to_plot = shap_values

    # SHAP summary plot for classification (using the selected SHAP values)
    shap.summary_plot(shap_values_to_plot, X, plot_type="bar")


if __name__ == '__main__':
    global_surrogate_regressor()