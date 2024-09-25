# Andrej Karpath 

## micrograd exercise

Building a tiny autograd (automatic gradient engine) with back propagation (reverse-mode autodiff) over a dynamically built DAG (directed acyclic graph). Source [GitHub] (https://github.com/karpathy/micrograd)

### Steps followed:

First 1 hr 25 mins of [Chapter#1] (https://www.youtube.com/watch?v=VMj-3S1tku0):

1. Create a small expression graph and calculate the expression using forward pass
2. Plot the expression graph
3. Enhance expression graph by adding labels and gradients to the nodes
4. Manually calculate the gradients using back propagation
5. Create a more complex expression graph and repeat above exercise
6. Build a backward function that can compute gradients at each node 
7. Use topological sort to auto compute gradients in reverse order