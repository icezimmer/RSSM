import torch
import torch.nn.functional as F

class Convolutional_LTI(torch.nn.Module):
    def __init__(self, eigs_A, norm_b, norm_c, value_d):
        super(Convolutional_LTI, self).__init__()
        self.eigs_A = eigs_A
        b = torch.rand(self.A.shape[0], 1, dtype=torch.float32) - 0.5
        self.b = b / torch.norm(b) * norm_b
        c = torch.rand(1, self.A.shape[0], dtype=torch.float32) - 0.5
        self.c = c / torch.norm(c) * norm_c
        self.d = torch.tensor([[value_d]], dtype=torch.float32)
        self.A_dsc, self.b_dsc = self.__discretize()

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
    
    def forward(self, u):
        """
        Predict all y(t). u has size [batch_size, input_size, seq_length], for now batch_size = 1 and input_size = 1
        """
        seq_length = u.size(2)
        eps = 1 / (seq_length-1)
        grid = eps*torch.arange(seq_length)

        f = torch.zeros((1,seq_length), dtype=torch.float32)
        j = 0
        for item in list(self.eigs_A.items()):
            alpha = item[0][0] # real part -> homothety exp(alpha*t)
            omega = item[0][1] # immaginary part -> rotation of theta = omega*t
            mul = item[1]

            for i in range(mul):
                f += 2 * eps * self.c[j] * self.b[j] * eps * torch.exp(alpha * grid) * torch.cos(omega * grid)
                j += 1 # j is the index of the current complex eigenvalue

        
        # f is my kernel [output_size, input_size, kernel_size]
        f = f.view(1,1,-1)
        y = F.conv1d(u, f)
        return y

