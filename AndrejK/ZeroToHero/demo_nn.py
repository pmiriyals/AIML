from nn import MLP
from torch_demo import MLPTorch
from engine import Value
import graph_viz

# Torch demo

xs = [2.0, 0.0]
ws = [-3.0, 1.0]
b = 6.8813735870195432

mlp_torch = MLPTorch(xs, ws, b)
print(mlp_torch)

# Scratch demo

# inputs x1, x2
x1 = Value(2.0, label='x1')
x2 = Value(0.0, label='x2')
# weights w1, w2 (synaptic strings for each input)
w1 = Value(-3.0, label='w1')
w2 = Value(1.0, label='w2')
# bias of the neuron
b = Value(6.8813735870195432, label='b')

# x1*w1 + x2*w2 + b
x1w1 = x1 * w1; x1w1.label = 'x1*w1'
x2w2 = x2 * w2; x2w2.label = 'x2*w2'
x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label='x1w1x2w2'
# cell body raw activation without the activation function (tanh)
n = x1w1x2w2 + b; n.label='n'
# --- 
e = (2*n).exp()
o = (e - 1) / (e + 1)
#o = n.tanh(); o.label = 'o' 
# ----
o.label = 'o'
o.backward()
dot = graph_viz.draw_dot(o)
dot.render('gout')
