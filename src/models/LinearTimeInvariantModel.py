import torch

class LinearTimeInvariantModel:
    def __init__(self, A, B, C, D):
        """
        Initialize the state-space model with system matrices A, B, C, D.
        These matrices define the dynamics of the system.
        """
        self.A = torch.tensor(A, dtype=torch.float32)
        self.B = torch.tensor(B, dtype=torch.float32)
        self.C = torch.tensor(C, dtype=torch.float32)
        self.D = torch.tensor(D, dtype=torch.float32)

    def update_state(self, x, u):
        """
        Update the state of the system.
        x is the current state, and u is the input.
        """
        return self.A @ x + self.B @ u

    def calculate_output(self, x):
        """
        Calculate the output of the system based on the current state.
        """
        return self.C @ x + self.D
