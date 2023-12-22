---
jupyter:
  kernelspec:
    display_name: thesis
    language: python
    name: python3
  language_info:
    codemirror_mode:
      name: ipython
      version: 3
    file_extension: .py
    mimetype: text/x-python
    name: python
    nbconvert_exporter: python
    pygments_lexer: ipython3
    version: 3.10.12
  nbformat: 4
  nbformat_minor: 2
---

::: {.cell .code execution_count="16"}
``` python
from src.models.LinearTimeInvariant import LinearTimeInvariant
import torch
```
:::

::: {.cell .code execution_count="17"}
``` python
# Example matrices for a simple LTI system
eigs_A = {(-1,1):1}
B = [[0.], [1.]]
C = [[1., 0.]]
D = [[0.]]

# Time step for discretization
dt = 0.1

# Create the LTIRNN model
lti_rnn = LinearTimeInvariant(eigs_A, B, C, D, dt)
```
:::

::: {.cell .code execution_count="18"}
``` python
# timesteps
N = 100

# Create a random input sequence
u = torch.randn(N, 1, 1)

# Initialize the state
x = torch.zeros(2, 1)

# Simulate the system N steps
for i in range(N):
    x, y_new = lti_rnn(x, u[i])
    # append the output to tensor
    if i == 0:
        y = y_new
    else:
        y = torch.cat((y, y_new), 0)
```
:::

::: {.cell .code execution_count="19"}
``` python
import matplotlib.pyplot as plt
# Plot the input
plt.plot(u[:,0].detach().numpy())
plt.show()

# Plot the output
plt.plot(y.detach().numpy())
plt.show()
```

::: {.output .display_data}
![](vertopal_7d5d97f8e0d14754982b37ce0e918ccf/e39f4c91e0ee69f7425c0b2b9e57e49ece88dfdc.png)
:::

::: {.output .display_data}
![](vertopal_7d5d97f8e0d14754982b37ce0e918ccf/10984b7792025114504d9aabb6235fa6339d413a.png)
:::
:::

::: {.cell .code execution_count="20"}
``` python
print(lti_rnn.A)
```

::: {.output .stream .stdout}
    tensor([[-1.,  1.],
            [-1., -1.]])
:::
:::