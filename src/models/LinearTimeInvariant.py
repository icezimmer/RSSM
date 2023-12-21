import torch

"""
Define a model subclass of torch.nn.Module that implements a linear time-invariant system.
"""
class LinearTimeInvariant(torch.nn.Module):
    def __init__(self, A, B, C, D, eps):
        """
        Initialize the state-space model with system matrices A, B, C, D.
        These matrices define the dynamics of the system.
        """
        super(LinearTimeInvariant, self).__init__()
        self.A = torch.tensor(A, dtype=torch.float32)
        self.B = torch.tensor(B, dtype=torch.float32)
        self.C = torch.tensor(C, dtype=torch.float32)
        self.D = torch.tensor(D, dtype=torch.float32)
        self.eps = torch.tensor([eps], dtype=torch.float32)

    def __discretize(self):
        """
        Discretize the system using the bilinear transform.
        """
        # A = (I - delta_t/2 * A)^-1 * (I + delta_t/2 * A)
        # B = (I - delta_t/2 * A)^-1 * delta_t * B
        # C = C
        # D = D
        I = torch.eye(self.A.shape[0], dtype=torch.float32)
        Ad = torch.inverse(I - self.eps/2 * self.A) @ (I + self.eps/2 * self.A)
        Bd = (self.eps * torch.inverse(I - self.eps/2 * self.A)) @  self.B
        return Ad, Bd

    def forward(self, x, u):
        """
        Compute the output of the system based on the current state.
        """
        Ad, Bd = self.__discretize()
        next_x = Ad @ x + Bd @ u
        y = self.C @ next_x + self.D @ u
        return next_x, y