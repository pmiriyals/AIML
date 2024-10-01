import torch

class MLPTorch:
    def __init__(self, xs, ws, b):
        self.xTs = []
        self.wTs = []

        for x, w in zip(xs, ws):
            xT = torch.Tensor([x]).double() ; xT.requires_grad = True ; self.xTs.append(xT)
            wT = torch.Tensor([w]).double() ; wT.requires_grad = True ; self.wTs.append(wT)

        bT = torch.Tensor([b]).double() ; bT.requires_grad = True

        self.out = self.__compute_nn(self.xTs, self.wTs, bT)
    
    def __compute_nn(self, xTs, wTs, bT):
        act = sum((x*w for x, w in zip(xTs, wTs)), bT)
        out = torch.tanh(act) # forward pass
        out.backward() # backward pass

        return out
    
    def __repr__(self):
        op = f'out.data = {self.out.data}\n______\n'
        op += ''
        count = 1
        for xT in self.xTs:
            op += f'x{count}.grad = {xT.grad}\n'
            count += 1

        count = 1
        for wT in self.wTs:
            op += f'w{count}.grad = {wT.grad}\n'
            count += 1
        return op
