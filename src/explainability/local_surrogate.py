import lime
import lime.lime_tabular
import pandas as pd
from src.data_processing.feature_engineering import data_feature_engineering


def local_surrogate_regressor(model, instance):
    df = pd.read_csv('/Users/carmine/PycharmProjects/CharlesHendrix/data/user_feedback.csv')
    df = data_feature_engineering(df, mode='fit')
    X = df.drop(columns=['UserInput'])

    explainer = lime.lime_tabular.LimeTabularExplainer(X.values,
                                                       feature_names=X.columns,
                                                       mode='regression',
                                                       discretize_continuous=True,
                                                       random_state=42)

    exp = explainer.explain_instance(
        instance.values[0],
        model.predict,
        num_features=10,
        num_samples=1000
    )

    print("Explanation as list of feature contributions:")
    print("Feature contributions to the prediction:")
    for condition, value in exp.as_list():
        direction = "increases" if value > 0 else "decreases"
        print(f"If {condition}, it {direction} the prediction by {abs(value):.4f}.")


