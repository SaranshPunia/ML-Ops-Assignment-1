# A1 – Boston Housing ML Workflow

This project demonstrates a simple ML workflow for predicting Boston housing prices using regression models. It includes data loading, preprocessing, training, and evaluation steps.

## Project Structure

- `misc.py`: Utility functions for data loading, splitting, pipeline building, and evaluation.
- `train.py`: Trains and evaluates a DecisionTreeRegressor.
- `train2.py`: Trains and evaluates a KernelRidge regressor.
- `requirements.txt`: Python dependencies.
- `.github/workflows/ci.yml`: GitHub Actions workflow for CI.

## Data

The Boston Housing dataset is loaded from [CMU StatLib](http://lib.stat.cmu.edu/datasets/boston) and cached locally.

## Usage

Install dependencies:

```sh
pip install -r requirements.txt
```

Run Decision Tree regression:

```sh
python train.py
```

Run Kernel Ridge regression:

```sh
python train2.py
```

## CI

On pushes to the `kernelridge` branch, GitHub Actions will automatically run both training scripts.
