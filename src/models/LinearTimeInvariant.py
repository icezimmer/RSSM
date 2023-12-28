import torch

"""
Define a model subclass of torch.nn.Module that implements a linear time-invariant system.
"""
class LinearTimeInvariant(torch.nn.Module):
    def __init__(self, eigs_A, B, C, D, eps):
        super(LinearTimeInvariant, self).__init__()
        self.A = self.__state_matrix(eigs_A)
        self.B = torch.tensor(B, dtype=torch.float32)
        self.C = torch.tensor(C, dtype=torch.float32)
        self.D = torch.tensor(D, dtype=torch.float32)
        self.eps = torch.tensor([eps], dtype=torch.float32)
        self.Ad, self.Bd = self.__discretize()  # Discretize once during initialization
        self.x = torch.zeros(self.A.shape[0], 1, dtype=torch.float32)

    def __state_matrix(self, eigs):
        """
        Compute the state matrix of the system.
        """
        A = torch.empty((0,0), dtype=torch.float32)
        
        for item in list(eigs.items()):
            alpha = item[0][0] # real part -> omotetia exp(alpha*t)
            omega = item[0][1] # immaginary part -> rotation of theta = omega*t
            mul = item[1]

            for i in range(mul):
                block = torch.tensor([[alpha, omega], [-omega, alpha]])
                A = torch.block_diag(A, block)
                
        return A
        

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

    def forward(self, u):
        """
        Compute the output of the system based on the current state.
        """
        self.x = self.Ad @ self.x + self.Bd @ u
        y = self.C @ self.x + self.D @ u
        return y