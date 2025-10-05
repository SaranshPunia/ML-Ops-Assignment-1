from __future__ import annotations
from sklearn.tree import DecisionTreeRegressor
from misc import load_data, split_features_target, train_and_evaluate

def main():   
    df = load_data()
    X_train, X_test, y_train, y_test = split_features_target(df, target_col='MEDV',
    test_size=0.2, random_state=42)
    model = DecisionTreeRegressor(random_state=42)
    mse = train_and_evaluate(model, with_scaler=False,
    X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    print(f"DecisionTreeRegressor Test MSE: {mse:.6f}")

if __name__ == "__main__":
    main()