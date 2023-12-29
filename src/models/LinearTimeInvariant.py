import torch

"""
Define a model subclass of torch.nn.Module that implements a linear time-invariant system.
"""
class LinearTimeInvariant(torch.nn.Module):
    def __init__(self, eigs_A, norm_b, norm_c, value_d, eps):
        super(LinearTimeInvariant, self).__init__()
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
        # b = (I - delta_t/2 * A)^-1 * delta_t * b
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
        Predict all y(t). u has size [seq_length, batch_size, input_size], for now batch_size = 1
        """
        T = u.size(0)
        for t in range(T):
            u_t = u[t]
            # transpose u_t to [input_size, 1]
            u_t = torch.transpose(u_t, 0, 1)
            y_t = self.__onestep(u_t)
            # transpose y_t to [1, output_size]
            y_t = torch.transpose(y_t, 0, 1)
            # unsqueeze y_t to [1, 1, output_size]
            y_t = torch.unsqueeze(y_t, 0)
            # stack the output
            if t == 0:
                y = y_t
            else:
                y = torch.cat((y, y_t), 0)
        return y