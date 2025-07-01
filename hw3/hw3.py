from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt

def load_and_center_dataset(filename):
    x = np.load(filename)
    n = len(x)
    mean = np.mean(x, axis = 0)
    x = x - mean
    return x

def get_covariance(dataset):
    n = dataset.shape[0]
    covariance = (1 / (n - 1)) * np.dot(dataset.T, dataset)
    return covariance

def get_eig(S, k):
    e_values, e_vectors = eigh(S)
    size = len(e_values)
    sorted_indices = np.argsort(e_values)[::-1]

    e_values_sorted = e_values[sorted_indices]
    e_vectors_sorted = e_vectors[:, sorted_indices]

    largest_eigenvalues = e_values_sorted[:k]
    largest_eigenvectors = e_vectors_sorted[:, :k]

    eigenvalues_matrix = np.diag(largest_eigenvalues)
    return eigenvalues_matrix, largest_eigenvectors

#I used AI to help with the math behind this function
def get_eig_prop(S, prop):
    e_values, e_vectors = eigh(S)

    # Sort eigenvalues in descending order and reorder eigenvectors accordingly
    sorted_indices = np.argsort(e_values)[::-1]
    e_values = e_values[sorted_indices]
    e_vectors = e_vectors[:, sorted_indices]

    # Compute cumulative variance proportion
    total_variance = np.sum(e_values)
    variance_ratios = e_values / total_variance

    # Select eigenvalues/eigenvectors based on the given proportion
    mask = variance_ratios > prop
    important_values = e_values[mask]
    important_vectors = e_vectors[:, mask]

    # Return diagonal matrix of eigenvalues and corresponding eigenvectors
    return np.diag(important_values), important_vectors


def project_and_reconstruct_image(image, U):
    product = np.matmul(U.T, image)
    x_pca = np.matmul(U, product)
    return x_pca

def display_image(im_orig_fullres, im_orig, im_reconstructed):
    # Please use the format below to ensure grading consistency
    fig, (ax1, ax2, ax3) = plt.subplots(figsize=(9,3), ncols=3)
    fig.tight_layout()

    im_orig_fullres = im_orig_fullres.reshape(218,178, 3)
    im_orig = im_orig.reshape(60,50)
    im_reconstructed = im_reconstructed.reshape(60,50)

    im1 = ax1.imshow(im_orig_fullres, aspect = 'equal')
    ax1.set_title("Original High Res")

    im2 = ax2.imshow(im_orig, aspect = 'equal')
    ax2.set_title("Original")

    im3 = ax3.imshow(im_reconstructed, aspect = 'equal')
    ax3.set_title("Reconstructed")

    fig.colorbar(im2, ax = ax2)
    fig.colorbar(im3, ax = ax3)

    plt.show()

    return fig, ax1, ax2, ax3

#I utilized AI to help with this function
def perturb_image(image, U, sigma):
    alpha = U.T @ image
    perturbed = np.random.normal(0, sigma, size=alpha.shape)
    alpha_perturbed = alpha + perturbed

    altered_image = U @alpha_perturbed
    return altered_image
