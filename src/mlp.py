# src/mlp.py

import torch
import torch.nn as nn

DROPOUT_RATE = 0.1


class MLP(nn.Module):
    """
    MLP (Multi-Layer Perceptron) class defining a fully
    connected neural network.

    This model implements a multi-layer perceptron with
    several linear layers, interleaved with ReLU activation
    functions and dropout layers to help prevent overfitting.
    It takes input vectors of size 16 and outputs vectors of
    size 3.

    The class also handles input conversion and reshaping
    within the `forward` method, ensuring compatibility with
    different input formats.

    Typical usage:
        model = MLP()
        output = model.forward(input_data)

    This architecture is suitable for classification or regression
    tasks with input dimension 16 and output dimension 3.
    """
    def __init__(self):
        """
        Initializes the MLP neural network architecture.

        This constructor sets up a multi-layer perceptron with
        four linear layers, interleaved with ReLU activation
        functions and dropout layers to prevent overfitting.
        The network takes input vectors of size 16 and outputs
        vectors of size 3.

        Args:
            None

        Returns:
            None

        Example:
            >>> model = MLP()
        """
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(16, 256),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
        )

    def forward(self, x):
        """
        Performs a forward pass through the network with the
        given input.

        This method ensures the input is a PyTorch tensor of
        type float32, reshapes it to match the network's expected
        input dimensions, passes it through the network, and then
        reshapes the output to a one-dimensional tensor.

        Args:
            x (torch.Tensor or array-like): The input data to
            process through the network.

        Returns:
            torch.Tensor: The network's output as a 1D tensor.

        Example:
            >>> output = model.forward([1.0, 2.0, 3.0])
            >>> print(output.shape)
            torch.Size([N])  # where N is the output dimension
            of the network
        """
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        x = x.view(1, -1)
        return self.net(x).view(-1)
