from nn import MLP
from engine import Value

def train_neural_net(mlp, xs, ys, steps):
    for k in range(steps):
        ypred = [mlp(x) for x in xs] # forward pass
        loss = sum((yout - ygt)**2 for yout, ygt in zip(ypred, ys)) # loss calc
        loss.backward() # compute gradients
        # adjust parameters
        for p in mlp.parameters():
            p.data -= 0.1 * p.grad
        # reset grad
        mlp.zero_grad()
        print(f'Step  = {k}, loss = {loss}')
    print(ypred)



x = [2.0, 3.0, -1.0]
n = MLP(3, [4, 4, 1])
print(n(x))

xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
] # 4 possible inputs to the neural net
ys = [1.0, -1.0, -1.0, 1.0]
train_neural_net(n, xs, ys, 20)