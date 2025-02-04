import os
import torch
import numpy as np
import os
import time
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.special import softmax
import matplotlib.pyplot as plt
import pandas as pd
import umap

from platt_scaling import platt_scaling_fit
from evidential import discounting_fit

MODES =  ["train", "val", "test"]


def l2_norm(X):
    # return the row-wise l2 norm of the vectors in X
    # X: (NxD) numpy array
    X_norm = np.linalg.norm(X, axis=1)
    return X / X_norm[:, np.newaxis]


def tsne(X, compress=True):
    """
    Returns 2D TNSE representation of the input
    Parameters
    ----------
    X : (NxD) numpy array
        Input data with N rows and feature dimension D
    compress : boolean, optional
        Whether the TSNE is completed with 50 randomly selected elements.
        This may speed up computation at the cost of accuracy. The default is True.
    Returns
    -------
    X_embedded : (Nx2) numpy array
        Dimensionality-reduced version of X
    """
    D_MAX = 50
    N, D = X.shape
    if compress and min(N, D) > D_MAX:
        X = PCA(n_components=D_MAX).fit_transform(X)
    X_embedded = TSNE(n_components=2, init="random").fit_transform(X)
    return X_embedded


def PCA_compress(X, dim=2):
    return PCA(n_components=2).fit_transform(X)
    
    
def umap_compress(X, compress=True):
    D_MAX = 50
    N, D = X.shape
    if compress and min(N, D) > D_MAX:
        X = PCA(n_components=D_MAX).fit_transform(X)
        
    reducer = umap.UMAP()
    X_embedded = reducer.fit_transform(X)
    return X_embedded


def resolve_save_dir(save_dir, experiment_name):
    # create novel save directory and experiment name based on existing names
    proposed_dir = os.path.join(save_dir, experiment_name)
    suffix = 0

    while os.path.exists(proposed_dir):
        print(f"Directory exists at {proposed_dir}, creating new directory")
        suffix += 1
        new_experiment_name = experiment_name + "_" + str(suffix)
        proposed_dir = os.path.join(save_dir, new_experiment_name)

    if suffix == 0:  # experiment name unchanged
        new_experiment_name = experiment_name

    os.makedirs(proposed_dir)
    print(f"Directory created at {proposed_dir}")
    return proposed_dir, new_experiment_name


def plot_emb(
    z, y, fig_path=None, protos=None, y_protos=None, title=None, compression="tsne"
):
    """
    Plot the embeddings of the samples and prototypes
    Parameters
    ----------
    z : (N, D) numpy array
        Embeddings vectors for both standard data and prototypes
    y : (N,) numpy array
        Integer labels for class
    fig_path : os.path, optional
        Path of the figure to save, if no path, figure does not save
    protos : (P, D) numpy array, optional
        Embedding vectors representing prototypes
    y_protos: (P,) numpy array, optional
        Integer labels for the class of prototypes
    title : String, optional
        Title of the plot. The default is None
    compression : String, optional
        Method of compression 'tsne'/'pca'. The default is 'tsne'
    Returns
    -------
    None.
    """
    N, D = z.shape
    
    max_num_pts = 2000
    # if N > 2000, we randomly plot 2000 points to reduce TSNE's computational load
    if N > max_num_pts:
        random_points = np.random.choice(range(N), max_num_pts, replace=False)
        z = z[random_points]
        y = y[random_points]
        
    if protos is not None:
        P, _ = protos.shape
        embeddings = np.concatenate((z, protos), axis=0)
    else:
        embeddings = z
    print(f"Plotting {D}->2 {compression} embedding, N={N}, N_visualized={len(embeddings)}.")

    if compression == "tsne":
        embeddings = tsne(embeddings)
    elif compression == "pca":
        embeddings = PCA_compress(embeddings)
    elif compression == "umap":
        embeddings = umap_compress(embeddings)
    else:
        raise NotImplementedError()

    if protos is not None:
        Pe = embeddings[-P:]
    Xe = embeddings[:N]

    fig, ax = plt.subplots(figsize=(8, 6))
    clr = np.array(["green", "orange", "red", "purple", "blue", "brown", "gray", "pink", "olive", "cyan"])
    # plot ordinary datapoints
    ax.scatter(x=Xe[:, 0], y=Xe[:, 1], s=20, c=clr[y], marker="o", alpha=0.2)
    # plot prototypical datapoints
    if protos is not None:
        ax.scatter(x=Pe[:, 0], y=Pe[:, 1], s=40, c=clr[y_protos], marker="x", alpha=1)
    if title is not None:
        fig.suptitle(title)
    if fig_path is not None:
        fig.savefig(fig_path)
    plt.close()


def save_csv(filename, y, pred, save_path):
    """
    Save the filename, associated label, and model predictions in CSV
    Parameters
    ----------
    filename : (N,) numpy array
        ndarray of strings as filenames
    y : (N,) numpy array
        Integer labels for class
    pred : (N, C) numpy array
        Float array of network outputs
    save_path : String, optional
        Path to save the CSV, if None, no CSV is saved
    Returns
    -------
    None.
    """
    data_dict = {"filename": filename, "y": y}
    N, C = pred.shape
    for c in range(C):
        data_dict["outputs_" + str(c)] = pred[:, c]
    df = pd.DataFrame.from_dict(data_dict)
    if save_path is not None:
        df.to_csv(save_path)
    return df
    

# helper for numpy array
def output_to_evidence(arr):
    return arr.clip(min=0)
        

def convert_history_to_pseudo(pred_history, label_bank, method="avg_pred", calibrate=True):
    """
    Converts prediction history from the agent to pseudolabels.
    Parameters
    ----------
    pred_history : dict
        pred_history is a dict with three keys as defined by MODES.
        pred_history[mode] is a dict where 
        keys are the unique ID (uid), 
        values are numpy arrays of network predictions over training epochs
    label_bank : dict
        label_bank keys are unique ID (uid) and values are the one-hot labels
    method : String, optional
        specifies the method used to produce pseudolabels, can be
        - avg_pred: averages the confidence [0..1] across all epochs
        - avg_logit: averages the logits across all epochs
        - evidence: averages the alpha from DST across all epochs
    calibrate : boolean, optional
        if calibrate, then utilize the validation set and label_bank to calibrate results
        with either temperature scaling (logits) or discount factor (evidential)
    Returns
    -------
    pseudo : dict
        Dictionary of pseudolabel values for the training set
        where keys are filenames and values are (C, ) numpy arrays 
        with values between [0..1] and sum = 1
    """
    pseudo = {}
    if method == "avg_pred":
        for k in pred_history["train"].keys():
            history = np.array(pred_history["train"][k]) # n_epoch x C
            confidence = softmax(history, axis=1)
            pseudo[k] = np.mean(confidence, axis=0)
    elif method == "avg_logit":
        # find the logit average for each example

        if calibrate: # perform temperature scaling    
            avg_va, y_va = [], []
            for k in pred_history["val"].keys():
                history = np.array(pred_history["val"][k]) # n_epoch x C
                avg_va.append(np.mean(history, axis=0))
                y_va.append(label_bank[k])
                
            temp = platt_scaling_fit(np.array(avg_va), y_va, num_iters=5000, mode="temp")
        else:
            temp = 1.0
        for k in pred_history["train"].keys():
            history = np.array(pred_history["train"][k]) # n_epoch x C
            pseudo[k] = softmax(np.mean(history, axis=0) * temp)
            
    elif method == "evidence":
        # find the alpha average per example
        
        if calibrate: # perform discounting    
            ev_va, y_va = [], []
            for k in pred_history["val"].keys():
                history = np.array(pred_history["val"][k]) # n_epoch x C
                history_evidence = output_to_evidence(history)
                ev_va.append(np.mean(history_evidence, axis=0))
                y_va.append(label_bank[k])
                
            dfs = [discounting_fit(np.array(ev_va), y_va, num_iters=5000) for _ in range(5)]
            df = np.mean(dfs)
        else:
            df = 1.0
        for k in pred_history["train"].keys():
            history = np.array(pred_history["train"][k]) # n_epoch x C
            history_evidence = output_to_evidence(history)
            discounted_evidence = np.mean(history_evidence, axis=0) * df # C
            alpha = discounted_evidence + 1
            pseudo[k] = alpha / np.sum(alpha)
    else:
        raise ValueError(f"method must be avg_pred/avg_logit/evidence, received {method}")
            
    return pseudo
    
def save_pseudolabels(pseudo, save_path):
    # save pseudolabels as a csv file
    data_dict = {"uid":[]}
    for k in pseudo.keys():
        data_dict["uid"].append(k)
        pseudolabel = pseudo[k]
        C = len(pseudolabel)
        for c in range(C):
            column_name = "pseudo_" + str(c)
            if data_dict.get(column_name) is None:
                data_dict[column_name] = []
            data_dict[column_name].append(pseudolabel[c])
    df = pd.DataFrame.from_dict(data_dict)
    if save_path is not None:
        df.to_csv(save_path)
    return df

if __name__ == "__main__":
    #resolve_save_dir("../logs/", "hello")
    history_test = {}
    history_test["train"] = {'a':[np.array([-3,4,2]), np.array([-1, 7, 2])], 
                            'b':[np.array([8,1,-1]), np.array([9, 0, -1])]}
    history_test["val"] = {'c':[np.array([-3,4,2]), np.array([-1, 7, 2])], 
                            'd':[np.array([8,1,-1]), np.array([9, 0, -1])]}
    labels = {'a':0, 'b':0, 'c':1, 'd':0}
    pseudo = convert_history_to_pseudo(history_test, labels, method='avg_pred', calibrate=False)
    print(pseudo)
    df = save_pseudolabels(pseudo, None)
    print(df)
