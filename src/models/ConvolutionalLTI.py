import torch
import torch.nn.functional as F

class ConvolutionalLTI(torch.nn.Module):
    def __init__(self, eigs_A, norm_b, norm_c):
        super(ConvolutionalLTI, self).__init__()
        self.eigs_A = eigs_A
        b = torch.rand(sum(eigs_A.values()), 1, dtype=torch.float32) - 0.5
        self.b = b / torch.norm(b) * norm_b
        c = torch.rand(1, sum(eigs_A.values()), dtype=torch.float32) - 0.5
        self.c = c / torch.norm(c) * norm_c
    
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
        y = F.conv1d(u, f) # +d*u, (for now d=0)
        return y

