import torch
import pytest
from regression import fit_regression_model


def get_train_data(dim=1):
    """
    dim is the number of features in the input. for our purposes it will be either 1 or 2.
    """
    X_2 = torch.tensor(
        [[24.,  2.],
         [24.,  4.],
         [16.,  3.],
         [25.,  6.],
         [16.,  1.],
         [19.,  2.],
         [14.,  3.],
         [22.,  2.],
         [25.,  4.],
         [12.,  1.],
         [24.,  7.],
         [19.,  1.],
         [23.,  7.],
         [19.,  5.],
         [21.,  3.],
         [16.,  6.],
         [24.,  5.],
         [19.,  7.],
         [14.,  4.],
         [20.,  3.]])
    y = torch.tensor(
        [[1422.4000],
         [1469.5000],
         [1012.7000],
         [1632.2000],
         [952.2000],
         [1117.7000],
         [906.2000],
         [1307.3000],
         [1552.8000],
         [686.7000],
         [1543.4000],
         [1086.5000],
         [1495.2000],
         [1260.7000],
         [1288.1000],
         [1111.5000],
         [1523.1000],
         [1297.4000],
         [946.4000],
         [1197.1000]])
    if dim == 1:
        X = X_2[:, :1]
    elif dim == 2:
        X = X_2
    else:
        raise ValueError("dim must be 1 or 2")
    return X, y


def test_fit_regression_model_1d():

    # print(torch.__version__)

    

    X, y = get_train_data(dim=1)
    model, loss = fit_regression_model(X, y)
    print(loss)

    assert loss.item() < 4321,  " loss too big"


def test_fit_regression_model_2d():
    X, y = get_train_data(dim=2)
    model, loss = fit_regression_model(X, y)
    assert loss.item() < 400


# def test_fit_and_predict_regression_model_1d():
#     X, y = get_train_data(dim=1)
#     print(f"Training data X: {X}")
#     print(f"Training data y: {y}")
    
#     model, loss = fit_regression_model(X, y)
#     print(f"Model: {model}")
#     print(f"Training loss: {loss}")
    
#     X_test = torch.tensor([[20.], [15.], [10.]])
#     y_pred = model(X_test)
#     print(f"Predictions: {y_pred}")
    
#     expected_values = torch.tensor([[1242.9958],
#         [ 932.9533],
#         [ 622.9108]])
#     print(f"Expected values: {expected_values}")
    
#     assert ((y_pred - expected_values).abs() < 2).all(), f"Prediction mismatch: {y_pred} != {expected_values}"
#     assert y_pred.shape == (3, 1)

def test_fit_and_predict_regression_model_1d():
    X, y = get_train_data(dim=1)
    model, loss = fit_regression_model(X, y)
    X_test = torch.tensor([[20.], [15.], [10.]])
    y_pred = model(X_test)
    assert ((y_pred - torch.tensor([[1242.9958],
        [ 932.9533],
        [ 622.9108]])).abs() < 2).all(), " y_pred is not correct"
    assert y_pred.shape == (3, 1), " y_pred shape is not correct"

def test_fit_and_predict_regression_model_2d():
    X, y = get_train_data(dim=2)
    model, loss = fit_regression_model(X, y)
    X_test = torch.tensor([[20., 2.], [15., 3.], [10., 4.]])
    y_pred = model(X_test)

    assert ((y_pred - torch.tensor([[1208.9059],
        [ 939.4942],
        [ 670.0824]])).abs() < 2).all(), " y_pred is not correct"
    assert y_pred.shape == (3, 1), " y_pred shape is not correct"


# def test_fit_and_predict_regression_model_2d():
#     X, y = get_train_data(dim=2)
#     print(f"Training data X: {X}")
#     print(f"Training data y: {y}")
    
#     model, loss = fit_regression_model(X, y)
#     print(f"Model: {model}")
#     print(f"Training loss: {loss}")
    
#     X_test = torch.tensor([[20., 2.], [15., 3.], [10., 4.]])
#     y_pred = model(X_test)
#     print(f"Predictions: {y_pred}")
    
#     expected_values = torch.tensor([[1208.9059],
#         [ 939.4942],
#         [ 670.0824]])
#     print(f"Expected values: {expected_values}")
    
#     assert ((y_pred - expected_values).abs() < 2).all(), f"Prediction mismatch: {y_pred} != {expected_values}"
#     assert y_pred.shape == (3, 1)


if __name__ == "__main__":
    test_fit_regression_model_1d()
    test_fit_regression_model_2d()
    test_fit_and_predict_regression_model_1d()
    test_fit_and_predict_regression_model_2d()