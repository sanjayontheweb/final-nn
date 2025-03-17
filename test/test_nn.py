# TODO: import dependencies and write unit tests below
import numpy as np
import pytest
from nn.nn import NeuralNetwork 
from nn.preprocess import sample_seqs, one_hot_encode_seqs

#Assisted by ChatGPT

def test_single_forward():
    # Initialize a simple neural network
    nn_arch = [
        {"input_dim": 3, "output_dim": 2, "activation": "relu"}
    ]
    nn = NeuralNetwork(nn_arch, lr=0.01, seed=42, batch_size=1, epochs=1, loss_function="mean_squared_error")
    
    # Create test inputs
    W = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])  # 2x3 matrix
    b = np.array([[0.1], [0.2]])  # 2x1 matrix
    A_prev = np.array([[1], [2], [3]])  # 3x1 matrix
    
    # Test ReLU activation
    A_curr, Z_curr = nn._single_forward(W, b, A_prev, "relu")
    
    # Expected Z = W·A + b
    expected_Z = np.array([[1.5], [3.4]])  # Manual calculation
    assert np.allclose(Z_curr, expected_Z), "Linear transformation (Z) is incorrect"
    
    # Expected A = ReLU(Z) = max(0, Z)
    expected_A = np.array([[1.5], [3.4]])  # All values are positive, so same as Z
    assert np.allclose(A_curr, expected_A), "ReLU activation is incorrect"
    
    # Test Sigmoid activation
    A_curr, Z_curr = nn._single_forward(W, b, A_prev, "sigmoid")
    
    # Expected A = sigmoid(Z)
    expected_A = 1 / (1 + np.exp(-expected_Z))
    assert np.allclose(A_curr, expected_A), "Sigmoid activation is incorrect"
    
    # Test invalid activation
    with pytest.raises(ValueError):
        nn._single_forward(W, b, A_prev, "invalid")

def test_forward():
    # Initialize a simple neural network with two layers
    nn_arch = [
        {"input_dim": 3, "output_dim": 2, "activation": "relu"},
        {"input_dim": 2, "output_dim": 1, "activation": "sigmoid"}
    ]
    nn = NeuralNetwork(nn_arch, lr=0.01, seed=42, batch_size=1, epochs=1, loss_function="mean_squared_error")
    
    # Create test input
    X = np.array([[1], [2], [3]])  # 3x1 input matrix
    
    # Perform forward pass
    output, cache = nn.forward(X)
    
    # Check output shape
    assert output.shape == (1, 1), "Output shape is incorrect"
    
    # Check cache contains expected keys
    expected_keys = {'A1', 'Z1', 'A2', 'Z2'}
    assert set(cache.keys()) == expected_keys, "Cache missing expected keys"
    
    # Check shapes of cached values
    assert cache['A1'].shape == (3, 1), "A1 shape incorrect"
    assert cache['Z1'].shape == (2, 1), "Z1 shape incorrect"
    assert cache['A2'].shape == (2, 1), "A2 shape incorrect"
    assert cache['Z2'].shape == (1, 1), "Z2 shape incorrect"
    
    # Check that values are within expected range
    assert np.all(output >= 0) and np.all(output <= 1), "Sigmoid output should be between 0 and 1"
    
    # For ReLU layer, check all values are non-negative
    assert np.all(cache['A1'] >= 0), "ReLU activation should be non-negative"

def test_single_backprop():
    # Initialize neural network
    nn_arch = [
        {"input_dim": 3, "output_dim": 2, "activation": "relu"}
    ]
    nn = NeuralNetwork(nn_arch, lr=0.01, seed=42, batch_size=1, epochs=1, loss_function="mean_squared_error")
    
    # Create test inputs
    W_curr = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])  # 2x3 matrix
    b_curr = np.array([[0.1], [0.2]])  # 2x1 matrix
    A_prev = np.array([[1.0], [2.0], [3.0]])  # 3x1 matrix
    Z_curr = np.array([[1.4], [3.2]])  # 2x1 matrix
    dA_curr = np.array([[0.2], [0.5]])  # 2x1 matrix
    
    # Test ReLU backprop
    dA_prev, dW_curr, db_curr = nn._single_backprop(
        W_curr, b_curr, Z_curr, A_prev, dA_curr, "relu"
    )
    
    # Check shapes
    assert dA_prev.shape == (3, 1), "dA_prev shape incorrect"
    assert dW_curr.shape == (2, 3), "dW_curr shape incorrect"
    assert db_curr.shape == (2, 1), "db_curr shape incorrect"
    
    # Test Sigmoid backprop
    dA_prev, dW_curr, db_curr = nn._single_backprop(
        W_curr, b_curr, Z_curr, A_prev, dA_curr, "sigmoid"
    )
    
    # Check shapes again
    assert dA_prev.shape == (3, 1), "dA_prev shape incorrect for sigmoid"
    assert dW_curr.shape == (2, 3), "dW_curr shape incorrect for sigmoid"
    assert db_curr.shape == (2, 1), "db_curr shape incorrect for sigmoid"
    
    # Test values are in reasonable range
    assert np.all(np.abs(dW_curr) < 1.0), "dW_curr values too large"
    assert np.all(np.abs(db_curr) < 1.0), "db_curr values too large"
    
    # Test invalid activation
    with pytest.raises(ValueError):
        nn._single_backprop(W_curr, b_curr, Z_curr, A_prev, dA_curr, "invalid")
    
    # Test zero gradient case
    dA_curr_zero = np.zeros((2, 1))
    dA_prev, dW_curr, db_curr = nn._single_backprop(
        W_curr, b_curr, Z_curr, A_prev, dA_curr_zero, "relu"
    )
    assert np.allclose(dW_curr, 0), "Zero gradient case failed for dW"
    assert np.allclose(db_curr, 0), "Zero gradient case failed for db"

def test_predict():
    # Initialize neural network
    nn_arch = [
        {"input_dim": 3, "output_dim": 2, "activation": "relu"}
    ]
    nn = NeuralNetwork(nn_arch, lr=0.01, seed=42, batch_size=1, epochs=1, loss_function="mean_squared_error")
    
    # Create test inputs with simple values
    W_curr = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])  # Simpler 2x3 matrix
    b_curr = np.array([[0.0], [0.0]])  # 2x1 matrix of zeros
    A_prev = np.array([[1.0], [2.0], [3.0]])  # 3x1 matrix
    Z_curr = np.dot(W_curr, A_prev) + b_curr  # Should be [[1.0], [2.0]]
    dA_curr = np.array([[1.0], [1.0]])  # Simple gradient
    
    # Test ReLU backprop with manual calculation
    dA_prev, dW_curr, db_curr = nn._single_backprop(
        W_curr, b_curr, Z_curr, A_prev, dA_curr, "relu"
    )
    
    # Manual calculations for ReLU
    manual_dZ = np.where(Z_curr > 0, dA_curr, 0)  # ReLU derivative
    manual_dW = np.dot(manual_dZ, A_prev.T) / A_prev.shape[1]
    manual_db = np.sum(manual_dZ, axis=1, keepdims=True) / A_prev.shape[1]
    manual_dA_prev = np.dot(W_curr.T, manual_dZ)
    
    # Compare with manual calculations
    assert np.allclose(dA_prev, manual_dA_prev), "dA_prev doesn't match manual calculation"
    assert np.allclose(dW_curr, manual_dW), "dW_curr doesn't match manual calculation"
    assert np.allclose(db_curr, manual_db), "db_curr doesn't match manual calculation"
    
    # Add print statements for debugging
    # print("Z_curr:", Z_curr)
    # print("manual_dZ:", manual_dZ)
    # print("Expected dW:", manual_dW)
    # print("Got dW:", dW_curr)
    # print("Expected db:", manual_db)
    # print("Got db:", db_curr)
    
    # Check shapes
    assert dA_prev.shape == (3, 1), f"dA_prev shape incorrect: {dA_prev.shape}"
    assert dW_curr.shape == (2, 3), f"dW_curr shape incorrect: {dW_curr.shape}"
    assert db_curr.shape == (2, 1), f"db_curr shape incorrect: {db_curr.shape}"


def test_binary_cross_entropy():
    nn_arch = [{"input_dim": 2, "output_dim": 1, "activation": "sigmoid"}]
    nn = NeuralNetwork(nn_arch, lr=0.01, seed=42, batch_size=1, epochs=1, loss_function="binary_cross_entropy")

    y = np.array([[1.0], [0.0]])
    y_hat = np.array([[0.8], [0.2]])
    loss = nn._binary_cross_entropy(y, y_hat)
    expected_loss = -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
    assert np.isclose(loss, expected_loss), "Loss calculation incorrect for realistic case"


def test_binary_cross_entropy_backprop():
    nn_arch = [{"input_dim": 2, "output_dim": 1, "activation": "sigmoid"}]
    nn = NeuralNetwork(nn_arch, lr=0.01, seed=42, batch_size=1, epochs=1, loss_function="binary_cross_entropy")

    y = np.array([[1.0], [0.0]])
    y_hat = np.array([[1.0], [0.0]])
    dA = nn._binary_cross_entropy_backprop(y, y_hat)
    assert not np.any(np.isinf(dA)), "Gradient contains infinite values"
    assert dA.shape == y.shape, "Gradient shape mismatch"

def test_mean_squared_error():
    nn_arch = [{"input_dim": 2, "output_dim": 1, "activation": "sigmoid"}]
    nn = NeuralNetwork(nn_arch, lr=0.01, seed=42, batch_size=1, epochs=1, loss_function="mean_squared_error")

    y = np.array([[1.0], [0.0]])
    y_hat = np.array([[0.8], [0.2]])
    loss = nn._mean_squared_error(y, y_hat)
    expected_loss = np.mean((y - y_hat) ** 2)
    assert np.isclose(loss, expected_loss), "Loss calculation incorrect for realistic case"

def test_mean_squared_error_backprop():
    nn_arch = [{"input_dim": 2, "output_dim": 1, "activation": "sigmoid"}]
    nn = NeuralNetwork(nn_arch, lr=0.01, seed=42, batch_size=1, epochs=1, loss_function="mean_squared_error")

    y = np.array([[1.0], [0.0]])
    y_hat = np.array([[1.0], [0.0]])
    dA = nn._mean_squared_error_backprop(y, y_hat)
    assert not np.any(np.isnan(dA)), "Gradient contains NaN values"
    assert not np.any(np.isinf(dA)), "Gradient contains infinite values"
    assert dA.shape == y.shape, "Gradient shape mismatch"

def test_sample_seqs():
    sequences = ['ATCG', 'GCTA', 'ATCG', 'GCTA', 'ATCG', 'GCTA']
    labels = [True, False, True, False, False, False]

    sampled_seqs, sampled_labels = sample_seqs(sequences, labels)
    assert len(sampled_seqs) == len(sampled_labels), "Sampled sequences and labels lengths do not match"
    assert sum(sampled_labels) == len(sampled_labels) // 2, "Sampled labels are not balanced"

def test_one_hot_encode_seqs():
    seq = ['AGA','TCT']
    expected_output = np.array([1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0])
    output = one_hot_encode_seqs(seq)
    assert np.array_equal(output, expected_output), "One-hot encoding is incorrect"