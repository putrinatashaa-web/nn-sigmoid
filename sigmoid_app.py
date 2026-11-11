import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_moons, make_circles
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ACTIVATION = "Sigmoid"
FUNCTIONS = {
    "ReLU": (nn.ReLU, lambda x: np.maximum(0, x), lambda x: (x > 0).astype(float), "f(x) = max(0, x)", "[0, infinity)", "ReLU is zero for negative inputs and increases linearly for positive inputs. Its derivative at zero is undefined; this demo displays zero there."),
    "Sigmoid": (nn.Sigmoid, lambda x: 1 / (1 + np.exp(-np.clip(x, -80, 80))), lambda x: (1 / (1 + np.exp(-np.clip(x, -80, 80)))) * (1 - 1 / (1 + np.exp(-np.clip(x, -80, 80)))), "f(x) = 1 / (1 + exp(-x))", "(0, 1)", "Sigmoid maps inputs between zero and one. Its gradients become small at extreme inputs."),
    "Tanh": (nn.Tanh, np.tanh, lambda x: 1 - np.tanh(x)**2, "f(x) = tanh(x)", "(-1, 1)", "Tanh is zero-centred, but gradients can become small at extreme inputs."),
}
activation_class, activation_fn, gradient_fn, formula, output_range, explanation = FUNCTIONS[ACTIVATION]

st.set_page_config(page_title=f"{ACTIVATION} Neural Network Lab", layout="wide")
st.title(f"{ACTIVATION} Activation Function & Neural Network")
st.write("An interactive PyTorch demonstration adapted from the lecturer's neural network classification example.")
st.info(explanation)

st.subheader("1. Explore the Activation Function")
st.write(f"**Formula:** `{formula}`  |  **Output range:** {output_range}")
selected_x = st.slider("Choose an input x", -8.0, 8.0, 1.0, 0.1)
x_values = np.linspace(-8, 8, 500)
y_values = activation_fn(x_values)
grad_values = gradient_fn(x_values)
value = float(activation_fn(np.array([selected_x]))[0])
grad = float(gradient_fn(np.array([selected_x]))[0])
left, right = st.columns(2)
with left:
    fig, ax = plt.subplots()
    ax.plot(x_values, y_values, label=ACTIVATION)
    ax.scatter([selected_x], [value], color="red", zorder=3)
    ax.axhline(0, color="gray", linewidth=0.6)
    ax.axvline(0, color="gray", linewidth=0.6)
    ax.set(xlabel="Input x", ylabel="Activation f(x)", title=f"{ACTIVATION} Curve")
    ax.grid(alpha=0.2)
    st.pyplot(fig)
    plt.close(fig)
with right:
    fig, ax = plt.subplots()
    ax.plot(x_values, grad_values, label="Gradient")
    ax.scatter([selected_x], [grad], color="red", zorder=3)
    ax.set(xlabel="Input x", ylabel="Derivative", title=f"{ACTIVATION} Gradient")
    ax.grid(alpha=0.2)
    st.pyplot(fig)
    plt.close(fig)
st.write(f"At x = **{selected_x:.2f}**, output = **{value:.4f}**, displayed gradient = **{grad:.4f}**")

st.sidebar.header("Neural Network Settings")
dataset_type = st.sidebar.selectbox("Dataset type", ["Moons", "Circles"])
n_samples = st.sidebar.slider("Number of samples", 200, 2000, 500, step=100)
noise = st.sidebar.slider("Noise level", 0.0, 0.5, 0.2, step=0.05)
test_size = st.sidebar.slider("Test size (fraction)", 0.1, 0.5, 0.3, step=0.05)
hidden_size = st.sidebar.slider("Hidden layer size", 4, 128, 16, step=4)
n_hidden_layers = st.sidebar.slider("Number of hidden layers", 1, 3, 2)
learning_rate = st.sidebar.select_slider("Learning rate", options=[0.001, 0.003, 0.01, 0.03, 0.1], value=0.01)
epochs = st.sidebar.slider("Epochs", 10, 500, 100, step=10)
batch_size = st.sidebar.slider("Batch size", 16, 256, 64, step=16)
seed = int(st.sidebar.number_input("Random seed", 0, 9999, 42))

@st.cache_data
def generate_data(dataset_type, n_samples, noise, test_size, seed):
    if dataset_type == "Moons":
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=seed)
    else:
        X, y = make_circles(n_samples=n_samples, noise=noise, factor=0.5, random_state=seed)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)
    return X_train, X_test, y_train, y_test

class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_hidden_layers):
        super().__init__()
        layers = []
        current_dim = input_dim
        for _ in range(n_hidden_layers):
            layers.extend([nn.Linear(current_dim, hidden_dim), activation_class()])
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, output_dim))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

def train_model(model, loader, epochs, learning_rate):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    history = []
    for _ in range(epochs):
        model.train()
        total_loss = 0.0
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_X), batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_X)
        history.append(total_loss / len(loader.dataset))
    return history

def accuracy(model, X, y):
    model.eval()
    with torch.no_grad():
        predicted = model(torch.tensor(X, dtype=torch.float32)).argmax(dim=1).numpy()
    return float(np.mean(predicted == y))

st.subheader("2. Generated Dataset")
X_train, X_test, y_train, y_test = generate_data(dataset_type, n_samples, noise, test_size, seed)
X_all = np.vstack([X_train, X_test])
y_all = np.concatenate([y_train, y_test])
col1, col2 = st.columns(2)
with col1:
    st.write("Sample of training data (first 5 rows):")
    st.dataframe({"x1": X_train[:5, 0], "x2": X_train[:5, 1], "y": y_train[:5]})
with col2:
    fig, ax = plt.subplots()
    ax.scatter(X_all[:, 0], X_all[:, 1], c=y_all, cmap="coolwarm", s=15)
    ax.set(xlabel="x1", ylabel="x2", title="Dataset by Class")
    st.pyplot(fig)
    plt.close(fig)

st.subheader("3. Train the Neural Network")
st.caption(f"The model uses {ACTIVATION} in every hidden layer. Its output layer produces two logits for CrossEntropyLoss.")
if st.button("Train Model", type="primary"):
    with st.spinner("Training the model..."):
        torch.manual_seed(seed)
        np.random.seed(seed)
        train_loader = torch.utils.data.DataLoader(
            SimpleDataset(X_train, y_train), batch_size=batch_size,
            shuffle=True, generator=torch.Generator().manual_seed(seed)
        )
        model = MLP(2, hidden_size, 2, n_hidden_layers)
        loss_history = train_model(model, train_loader, epochs, learning_rate)
        train_acc = accuracy(model, X_train, y_train)
        test_acc = accuracy(model, X_test, y_test)
    st.session_state["results"] = dict(
        key=(ACTIVATION, dataset_type, n_samples, noise, test_size, hidden_size,
             n_hidden_layers, learning_rate, epochs, batch_size, seed),
        model=model, loss_history=loss_history, train_acc=train_acc, test_acc=test_acc
    )

key = (ACTIVATION, dataset_type, n_samples, noise, test_size, hidden_size,
       n_hidden_layers, learning_rate, epochs, batch_size, seed)
results = st.session_state.get("results")
if results and results["key"] == key:
    col_loss, col_metrics = st.columns(2)
    with col_loss:
        fig, ax = plt.subplots()
        ax.plot(range(1, epochs + 1), results["loss_history"])
        ax.set(xlabel="Epoch", ylabel="Cross-entropy loss", title="Training Loss")
        st.pyplot(fig)
        plt.close(fig)
    with col_metrics:
        st.metric("Train Accuracy", f"{results['train_acc'] * 100:.2f}%")
        st.metric("Test Accuracy", f"{results['test_acc'] * 100:.2f}%")
        st.caption("A large train-test accuracy gap can indicate overfitting.")
    st.subheader("4. Decision Boundary")
    x_min, x_max = X_all[:, 0].min() - 0.5, X_all[:, 0].max() + 0.5
    y_min, y_max = X_all[:, 1].min() - 0.5, X_all[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
    model = results["model"]
    model.eval()
    with torch.no_grad():
        predictions = model(grid).argmax(dim=1).numpy().reshape(xx.shape)
    fig, ax = plt.subplots()
    ax.contourf(xx, yy, predictions, alpha=0.3, cmap="coolwarm")
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap="coolwarm", edgecolors="black", s=20)
    ax.set(xlabel="x1", ylabel="x2", title=f"{ACTIVATION} Decision Boundary (Test Points)")
    st.pyplot(fig)
    plt.close(fig)
    st.caption("The background shows model predictions; dots show held-out test samples.")
else:
    st.info("Choose settings and click Train Model. If you change a setting, train again to update results.")
