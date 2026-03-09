# Two Moons Classification Engine

This project demonstrates a classic implementation of a high-performance neural network built from scratch in Rust and interfaced with Python. It solves the non-linear "Two Moons" classification problem by warping input space through hidden layers.

> This is Part 2 of the [neural_engine](https://github.com/elvert19/neural_engine) project.

---

## The Mathematical Model

The engine solves the classification task by performing non-linear coordinate transformations:

- **Forward Pass:** Computes `Z = XW + b` followed by a `tanh` activation, which "folds" the 2D plane to make the interlocking moons linearly separable.
- **Activation:** Uses `tanh` in the hidden layer to introduce non-linearity, and the sigmoid function `σ(z) = 1 / (1 + e^−z)` in the output layer to map results into a [0, 1] probability range.
- **Optimization:** Uses the Adam Optimizer to adapt the learning rate, minimizing Mean Squared Error (MSE) loss:

$$\text{Loss} = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2$$

---

## Findings & Analysis

### 1. The Power of Non-Linearity

A linear model would be completely incapable of separating the Two Moons dataset because the crescents are interlocking. By using a 16-neuron hidden layer with the `tanh` activation function, the model performs a non-linear mapping — transforming the 2D input space **R²** into a 16-dimensional hidden space **R¹⁶**, where the "fold" created by `tanh` allows for a clean hyperplane separation.

### 2. Convergence Behavior (Adam Optimizer)

The model exhibits rapid convergence, typically reaching an MSE near `0.0000` within 200–500 epochs.

- **Why it works:** The Adam Optimizer dynamically adjusts the learning rate `α` for each parameter by maintaining estimates of both the first and second moments of the gradients. This prevents the model from getting stuck in saddle points — areas where the gradient is flat but the loss is still high.
- **Loss decay:** The loss curve follows an exponential decay, characteristic of high-performing backpropagation engines.

### 3. Classification Boundaries

The `plot_decision_boundary` function reveals that the engine learns a curved boundary that perfectly mirrors the gap between the two moons. This confirms that:

- The **Forward Pass** correctly propagates features through the dense layers.
- The **Backward Pass** accurately calculates the partial derivatives `∂Loss/∂W` to tune the weights, ensuring the boundary is optimized across all 500 samples.

### 4. Hardware/Software Synergies

By moving compute-intensive matrix operations to Rust, execution times are significantly faster than pure Python implementations. The use of `ndarray` for tensor manipulation ensures efficient memory allocation — critical when scaling from 500 samples (Moons) to 60,000 samples (MNIST).

---

## Setup & Installation

To run this project, the Neural Engine must be compiled locally as a Python module.

### Step 1 — Clone the Engine

Clone the core Rust library:

```bash
git clone https://github.com/elvert19/neural_engine.git
cd neural_engine
```

### Step 2 — Build and Install

Compile the Rust engine into a Python-importable module using `maturin`:

```bash
maturin develop --release
```

### Step 3 — Run the Moons Project

Navigate to the `moon_project` folder and run the visualization:

```bash
cd ../moon_project
python3 moon.py
```

---

## Project Structure

```
.
├── moon.py              # Python test harness: generates data, runs the Rust model, and visualizes the decision boundary
└── neural_engine/       # External Rust dependency: core matrix math, backpropagation, and optimizer logic
```

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'neural_engine'`**  
Ensure you are in the same virtual environment where you ran `maturin develop`.

**Visualization window doesn't open on Linux**  
Install the required backend:

```bash
sudo apt-get install python3-tk
```
