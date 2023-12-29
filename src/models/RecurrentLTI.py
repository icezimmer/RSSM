import torch

"""
Define a model subclass of torch.nn.Module that implements a linear time-invariant system.
"""
class RecurrentLTI(torch.nn.Module):
    def __init__(self, eigs_A, norm_b, norm_c, value_d, eps):
        super(RecurrentLTI, self).__init__()
        self.A = self.__state_matrix(eigs_A)
        b = torch.rand(self.A.shape[0], 1, dtype=torch.float32) - 0.5
        self.b = b / torch.norm(b) * norm_b
        c = torch.rand(1, self.A.shape[0], dtype=torch.float32) - 0.5
        self.c = c / torch.norm(c) * norm_c
        self.d = torch.tensor([[value_d]], dtype=torch.float32)
        self.eps = torch.tensor([eps], dtype=torch.float32)
        self.A_dsc, self.b_dsc = self.__discretize()  # Discretize once during initialization
        self.x = torch.zeros(self.A.shape[0], 1, dtype=torch.float32)

    def __state_matrix(self, eigs):
        """
        Compute the state matrix of the system.
        """
        A = torch.empty((0,0), dtype=torch.float32)
        
        for item in list(eigs.items()):
            alpha = item[0][0] # real part -> homothety of exp(alpha*t)
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
        # A = (I - eps/2 * A)^-1 * (I + eps/2 * A)
        # b = (I - eps/2 * A)^-1 * eps * b
        # c = c
        # d = d
        I = torch.eye(self.A.shape[0], dtype=torch.float32)
        A_dsc = torch.inverse(I - self.eps/2 * self.A) @ (I + self.eps/2 * self.A)
        b_dsc = (self.eps * torch.inverse(I - self.eps/2 * self.A)) @  self.b
        return A_dsc, b_dsc

    def __onestep(self, u):
        """
        Compute the output of the system based on the current state.
        u has size [input_size, 1]
        """
        self.x = self.A_dsc @ self.x + self.b_dsc @ u
        y = self.c @ self.x + self.d @ u
        return y
    
    def forward(self, u):
        """
        Predict all y(t). u has size [seq_length, batch_size, input_size], for now batch_size = 1 and input_size = 1
        """
        seq_length = u.size(0)
        for k in range(seq_length):
            u_k = u[k]
            # transpose u_k to [input_size, 1]
            u_k = torch.transpose(u_k, 0, 1)
            y_k = self.__onestep(u_k)
            # transpose y_k to [1, output_size]
            y_k = torch.transpose(y_k, 0, 1)
            # unsqueeze y_k to [1, 1, output_size]
            y_k = torch.unsqueeze(y_k, 0)
            # stack the output
            if k == 0:
                y = y_k
            else:
                y = torch.cat((y, y_k), 0)
        return y