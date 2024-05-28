import torch

X_orig = torch.arange(-5, 5,0.1)
m_orig = 500
b_orig = -5
error = torch.rand(len(X_orig))
Y_orig = m_orig * X_orig + b_orig + error

m_hat = torch.tensor(0., requires_grad=True)
b_hat = torch.tensor(0., requires_grad=True)
steps = 10000
lr = 0.01

for i in range(steps):
    Y_pred = m_hat * X_orig + b_hat
    loss = torch.mean((Y_orig-Y_pred)**2)
    loss.backward()
    with torch.no_grad():
        m_hat -= lr * m_hat.grad
        b_hat -= lr * b_hat.grad
    m_hat.grad.zero_()
    b_hat.grad.zero_()
    if i % 1000 == 0:
        print(f"{loss=}, {m_hat=}, {b_hat=}")
