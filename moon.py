import neural_engine
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons


print("Generating Two Moons data...")

X, y = make_moons(n_samples=500, noise=0.1, random_state=42)

# Your Rust engine expects y to be a 2D column vector: [[0], [1], ...]
y_train = y.reshape(-1, 1).astype(np.float64)
X_train = X.astype(np.float64)

print("Building model in Rust...")
model = neural_engine.Sequential()

# Layer 1: 2 inputs (X, Y) -> 16 hidden neurons
model.add_dense(2, 16)
model.add_tanh()

# Layer 2: 16 hidden -> 1 output neuron
model.add_dense(16, 1)
model.add_sigmoid()

# --- 3. Setup Optimizer & Loss ---
# Using 'adam' for faster convergence and 'mse' as requested
model.set_optimizer("adam", 0.05)
model.set_loss("mse")

# --- 4. Train ---
print("Starting training...")
# This calls your Rust .train() method
model.train(X_train, y_train, epochs=1000)

# --- 5. Visualization ---
def plot_decision_boundary(model, X, y):
    print("Generating decision boundary plot...")
    # Create a grid of points to "color in" the background
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    
    # Flatten the grid and predict every point
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    # This calls my Rust .predict() method
    predictions = model.predict(grid_points)
    
    # Reshape back to the grid shape for plotting
    Z = predictions.reshape(xx.shape)

    # Plot the contours (the colored background)
    plt.contourf(xx, yy, Z, levels=[0, 0.5, 1], cmap="RdBu", alpha=0.3)
    
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="RdBu", edgecolors='k')
    plt.title("Neural Engine v1: Two Moons Classification")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.show()

plot_decision_boundary(model, X, y)