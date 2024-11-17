import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
import os
from scipy.spatial.distance import cdist
import matplotlib
matplotlib.use('Agg')  # Fix threading issues for Flask

# Define the directory to save results
result_dir = "results"
os.makedirs(result_dir, exist_ok=True)

def generate_ellipsoid_clusters(distance, n_samples=100, cluster_std=0.5):
    np.random.seed(0)
    covariance_matrix = np.array([[cluster_std, cluster_std * 0.8], 
                                   [cluster_std * 0.8, cluster_std]])
    
    # Generate the first cluster (class 0)
    X1 = np.random.multivariate_normal(mean=[1, 1], cov=covariance_matrix, size=n_samples)
    y1 = np.zeros(n_samples)

    # Generate the second cluster (class 1)
    X2 = np.random.multivariate_normal(mean=[1, 1], cov=covariance_matrix, size=n_samples)
    X2[:, 0] += distance  # Shift along x-axis
    X2[:, 1] += distance  # Shift along y-axis
    y2 = np.ones(n_samples)

    # Combine clusters
    X = np.vstack((X1, X2))
    y = np.hstack((y1, y2))
    return X, y

def fit_logistic_regression(X, y):
    model = LogisticRegression()
    model.fit(X, y)
    beta0 = model.intercept_[0]
    beta1, beta2 = model.coef_[0]
    return model, beta0, beta1, beta2

def do_experiments(start, end, step_num):
    shift_distances = np.linspace(start, end, step_num, endpoint=True)
    beta0_list, beta1_list, beta2_list = [], [], []
    slope_list, intercept_list, loss_list, margin_widths = [], [], [], []

    # Ensure results directory is cleared
    for file in os.listdir(result_dir):
        os.remove(os.path.join(result_dir, file))

    # First Plot: Dataset Visualization
    n_cols = 3
    n_rows = (step_num + n_cols - 1) // n_cols  # Adjust rows to accommodate all steps
    plt.figure(figsize=(20, n_rows * 6))

    for i, distance in enumerate(shift_distances, 1):
        X, y = generate_ellipsoid_clusters(distance=distance)
        model, beta0, beta1, beta2 = fit_logistic_regression(X, y)

        # Store parameters for later plots
        beta0_list.append(beta0)
        beta1_list.append(beta1)
        beta2_list.append(beta2)
        slope_list.append(-beta1 / beta2)
        intercept_list.append(-beta0 / beta2)
        loss_list.append(log_loss(y, model.predict_proba(X)))

        # Create subplot
        plt.subplot(n_rows, n_cols, i)
        plt.scatter(X[y == 0, 0], X[y == 0, 1], color='blue', alpha=0.6, label='Class 0')
        plt.scatter(X[y == 1, 0], X[y == 1, 1], color='red', alpha=0.6, label='Class 1')
        
        # Plot decision boundary and confidence areas
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
        Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1].reshape(xx.shape)
        
        plt.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)
        plt.contourf(xx, yy, Z, levels=[0.0, 0.7], alpha=0.2, colors=['blue'])
        plt.contourf(xx, yy, Z, levels=[0.3, 1.0], alpha=0.2, colors=['red'])

        # Calculate margin width using distances between confidence contours
        class_1_contour = plt.contour(xx, yy, Z, levels=[0.7], colors=['red'], alpha=0)
        class_0_contour = plt.contour(xx, yy, Z, levels=[0.3], colors=['blue'], alpha=0)

        # Extract paths for margin width calculation
        try:
            vertices_1 = class_1_contour.collections[0].get_paths()[0].vertices
            vertices_0 = class_0_contour.collections[0].get_paths()[0].vertices
            distances = cdist(vertices_1, vertices_0)
            margin_width = np.min(distances)
            margin_widths.append(margin_width)
        except IndexError:
            margin_widths.append(0)  # If contours are not present, append 0

        plt.title(f"Shift Distance = {distance:.2f}", fontsize=12)
        plt.xlabel("x1")
        plt.ylabel("x2")

    plt.tight_layout()
    plt.savefig(f"{result_dir}/dataset.png")
    plt.close()

    # Second Plot: Parameters vs Shift Distance
    plt.figure(figsize=(20, 20))

    # Beta0 plot
    plt.subplot(3, 3, 1)
    plt.plot(shift_distances, beta0_list, marker='o', color='blue')
    plt.title("Shift Distance vs Beta0")
    plt.xlabel("Shift Distance")
    plt.ylabel("Beta0")

    # Beta1 plot
    plt.subplot(3, 3, 2)
    plt.plot(shift_distances, beta1_list, marker='o', color='orange')
    plt.title("Shift Distance vs Beta1 (Coefficient for x1)")
    plt.xlabel("Shift Distance")
    plt.ylabel("Beta1")

    # Beta2 plot
    plt.subplot(3, 3, 3)
    plt.plot(shift_distances, beta2_list, marker='o', color='green')
    plt.title("Shift Distance vs Beta2 (Coefficient for x2)")
    plt.xlabel("Shift Distance")
    plt.ylabel("Beta2")

    # Slope plot
    plt.subplot(3, 3, 4)
    plt.plot(shift_distances, slope_list, marker='o', color='purple')
    plt.title("Shift Distance vs Beta1 / Beta2 (Slope)")
    plt.xlabel("Shift Distance")
    plt.ylabel("Slope")

    # Intercept plot
    plt.subplot(3, 3, 5)
    plt.plot(shift_distances, intercept_list, marker='o', color='brown')
    plt.title("Shift Distance vs Beta0 / Beta2 (Intercept)")
    plt.xlabel("Shift Distance")
    plt.ylabel("Intercept")

    # Logistic loss plot
    plt.subplot(3, 3, 6)
    plt.plot(shift_distances, loss_list, marker='o', color='red')
    plt.title("Shift Distance vs Logistic Loss")
    plt.xlabel("Shift Distance")
    plt.ylabel("Logistic Loss")

    # Margin width plot
    plt.subplot(3, 3, 7)
    plt.plot(shift_distances, margin_widths, marker='o', color='pink')
    plt.title("Shift Distance vs Margin Width")
    plt.xlabel("Shift Distance")
    plt.ylabel("Margin Width")

    plt.tight_layout()
    plt.savefig(f"{result_dir}/parameters_vs_shift_distance.png")
    plt.close()

if __name__ == "__main__":
    start = 0.25
    end = 2.0
    step_num = 8
    do_experiments(start, end, step_num)