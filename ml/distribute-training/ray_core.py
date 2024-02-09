import time
from operator import itemgetter

import pandas as pd
import ray
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import zipfile
import numpy as np


# Set number of models to train
NUM_MODELS = 20

# Initialize Ray runtime


# Prepare dataset
# X, y = fetch_california_housing(data_home="~/scikit_learn_data",download_if_missing=False,return_X_y=True, as_frame=True)



# Load the data from the extracted files into variables x and y
# For example, if the data is saved as NumPy arrays in separate files named 'x.npy' and 'y.npy'



# # # Put data in the object store
# X_train_ref = ray.put(X_train)
# X_test_ref = ray.put(X_test)
# y_train_ref = ray.put(y_train)
# y_test_ref = ray.put(y_test)


# Implement function to train and score model
@ray.remote
def train_and_score_model(
    # train_set_ref: pd.DataFrame,
    # test_set_ref: pd.DataFrame,
    # train_labels_ref: pd.Series,
    # test_labels_ref: pd.Series,
    n_estimators: int,
) -> tuple[int, float]:
    X, y = fetch_california_housing(data_home="~/scikit_learn_data",download_if_missing=False,return_X_y=True, as_frame=True)

    train_set_ref, test_set_ref, train_labels_ref, test_labels_ref = train_test_split(
    X, y, test_size=0.2, random_state=201
    )

    start_time = time.time()  # measure wall time for single model training

    model = RandomForestRegressor(n_estimators=n_estimators, random_state=201)
    model.fit(train_set_ref, train_labels_ref)
    y_pred = model.predict(test_set_ref)
    score = mean_squared_error(test_labels_ref, y_pred)

    time_delta = time.time() - start_time
    print(
        f"n_estimators={n_estimators}, mse={score:.4f}, took: {time_delta:.2f} seconds"
    )
    return n_estimators, score


# Implement function that runs parallel model training
def run_parallel(n_models: int) -> list[tuple[int, float]]:
    results_ref = [
        train_and_score_model.remote(
            # train_set_ref=X_train_ref,
            # test_set_ref=X_test_ref,
            # train_labels_ref=y_train_ref,
            # test_labels_ref=y_test_ref,
            n_estimators=8 + 4 * j,
        )
        for j in range(n_models)
    ]
    return ray.get(results_ref)


# Run parallel model training
if __name__ == "__main__":
    mse_scores = run_parallel(n_models=NUM_MODELS)

    # Analyze results
    best = min(mse_scores, key=itemgetter(1))
    print(f"Best model: mse={best[1]:.4f}, n_estimators={best[0]}")

    # Shutdown Ray runtime
    ray.shutdown()