**Two Moons Classification Engine**
This project demonstrates a  classic  implementation of a high-performance Neural Network, but this time it is built from scratch in Rust and interfaced with Python. It solves the non-linear "Two Moons" classification problem by warping input space through hidden layers.

This is part 2 of the first project,[neural_engine](https://github.com/elvert19/neural_engine)

**The Mathematical Model**

The engine solves the classification task by performing non-linear coordinate transformations:

Forward Pass: The engine computes $Z = XW + b$ followed by a $\tanh$ activation, which "folds" the 2D plane to make the interlocking moons linearly separable.

Activation: We use $\tanh$ in the hidden layer to introduce non-linearity and $\sigma(z) = \frac{1}{1+e^{-z}}$ in the output layer to map results into a $[0, 1]$ probability range.

Optimization: The model uses the Adam Optimizer to adapt the learning rate, minimizing the Mean Squared Error (MSE) loss:$Loss = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2$

 
**Findings & Analysis**


1. _The Power of Non-Linearity_. Our findings demonstrate that a linear model would be completely incapable of separating the Two Moons dataset because the crescents are interlocking. By utilizing a 16-neuron hidden layer combined with the $\tanh$ activation function, the model effectively performs a non-linear mapping:
Mathematically, the engine transforms the 2D input space $\mathbb{R}^2$ into a 16-dimensional hidden space $\mathbb{R}^{16}$, where the "fold" created by the $\tanh$ activation allows for a clean hyperplane separation.


2. _Convergence Behavior_ (Adam Optimizer)The model exhibits rapid convergence, typically reaching an MSE of near 0.0000 within 200–500 epochs. Why it works: The Adam Optimizer dynamically adjusts the learning rate $\alpha$ for each parameter by maintaining estimates of both the first and second moments of the gradients. This prevents the model from getting stuck in "saddle points"—areas where the gradient is flat but the loss is still high. Loss Decay: The loss curve follows an exponential decay, characteristic of high-performing backpropagation engines. 


3. _Classification Boundaries_. The plot_decision_boundary function reveals that the engine learns a "curved" boundary that perfectly mirrors the gap between the two moons. This confirms that: The Forward Pass correctly propagates features through the dense layers. The Backward Pass accurately calculates the partial derivatives $\frac{\partial Loss}{\partial W}$ to tune the weights, ensuring the boundary is optimized across all 500 samples.


4._Hardware/Software Synergies_. By moving the compute-intensive matrix operations to Rust, we achieved execution times significantly faster than pure Python implementations. The use of ndarray for tensor manipulation ensures efficient memory allocation, which is critical when scaling from 500 samples (Moons) to 60,000 samples (MNIST).



To run this project, you must have the Neural Engine compiled locally as a Python module.
**1. Clone the Engine**

First, clone the core Rust library:
Bash
git clone https://github.com/elvert19/neural_engine.git
cd neural_engine


2. **Build and Install**
Compile the Rust engine into a Python-importable module using maturin:
Bash
maturin develop --release


4. **Run the Moons Project**
Now, navigate to your moon_project folder and run the visualization:
Bash
cd ../moon_project
python3 moon.py

 
 **Project Structure**
    • moon.py: The Python test harness that generates data, initiates the Rust Sequential model, and visualizes the decision boundary using matplotlib.
    • neural_engine/: The external Rust dependency containing the core matrix math, backpropagation, and optimizer logic.
    
 Troubleshooting
    • Missing Module: If you see ModuleNotFoundError: No module named 'neural_engine', ensure you are in the same virtual environment where you ran maturin develop.
    • Visualization Window: If plt.show() does not open a window on Linux, install the backend with: sudo apt-get install python3-tk.

   



