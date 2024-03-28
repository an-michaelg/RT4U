import numpy as np
from sklearn.model_selection import train_test_split
from platt_scaling import platt_scaling_fit
from scipy.special import softmax
from conformal_pred_algos import *
from sklearn.metrics import balanced_accuracy_score

# test the coverage of conformal prediction
def coverage_test(y, preds_cover):
    # y is (N,) integer array of C classes
    # preds_cover is (N, C) boolean array indicating the cover
    correct = []
    for i in range(len(y)):
        if preds_cover[i, y[i]]:
            correct.append(True)
        else:
            correct.append(False)
    return np.mean(np.array(correct))
    
def class_conditional_coverage_test(y, preds_cover):
    # returns an accuracy for each class
    N, num_classes = preds_cover.shape
    class_cond_acc = np.zeros(num_classes)
    for c in range(num_classes):
        mask = y == c
        preds_c = preds_cover[mask]
        N_c = len(preds_c)
        correct = 0
        for i in range(N_c):
            if preds_c[i,c]:
                correct += 1
        class_cond_acc[c] = correct/N_c
    return class_cond_acc
    
def balanced_coverage_test(y, preds_cover):
    return np.mean(class_conditional_coverage_test(y, preds_cover))
    
def cardinality_test(preds_cover):
    cardinality = np.sum(preds_cover, axis=1)
    return np.mean(cardinality)
    
def adaptivity_test(y, preds_cover):
    N, C = preds_cover.shape
    cardinality = np.sum(preds_cover, axis=1)
    correct = coverage_test(y, preds_cover)
    card_correct = cardinality[correct]
    card_incorrect = cardinality[~correct]
    
    #n_bins = n_classes + 1
    #c_amount, c_bins = np.histogram(card_correct, bins=np.arange(n_bins)+0.5)
    #n_amount, _ = np.histogram(card_incorrect, bins=np.arange(n_bins)+0.5)
    #card_acc = c_amount / (c_amount + n_amount + 1e-9)
    #print(f"Number of correct preds of size [1..N_classes]: {c_amount}")
    #print(f"Number of incorrect preds of size [1..N_classes]: {n_amount}")
    #print(f"Accuracy of preds of size [1..N_classes]: {card_acc}")

    top1_accs = []
    pred_classes = np.argmax(preds, axis=1)
    for i in np.arange(n_classes)+1:
        gt_w_card_i = y[cardinality==i]
        pred_w_card_i = pred_classes[cardinality==i]
        acc_w_card_i = balanced_accuracy_score(gt_w_card_i, pred_w_card_i)#np.sum(gt_w_card_i == pred_w_card_i) / (len(gt_w_card_i) + 1e-9)
        top1_accs.append(acc_w_card_i)
    print(f"Top-1 acc of preds of size [1..N_classes]: {top1_accs}")
    return top1_accs

def is_increasing_by_one(arr):
    if len(arr) <= 1:
        return True  # A single element or empty array is always considered increasing
    
    for i in range(1, len(arr)):
        if arr[i] != arr[i - 1] + 1:
            return False
    return True
        
def ordinality_test(a):
    true_indices = [i for i in range(len(a)) if a[i] == True]
    return is_increasing_by_one(true_indices)
    
def ordinality_test_arr(arr):
    ordinal = []
    for i in range(arr.shape[0]):
        ordinal.append(ordinality_test(arr[i]))
    return np.mean(ordinal)
    
# a method of evaluating the distribution of conformal prediction
# it performs X number of random splits to calibration and test sets
# and conformal prediction using each set
def develop_coverage_distribution_logits(y, logits, n_splits=100, test_size=0.5, alpha=0.1, calibrate=True):
    
    cov, bcov, card = [], [], []
    for i in range(n_splits):
        val_logits, test_logits, y_val, y_test = train_test_split(logits, y, test_size=test_size)
        
        if calibrate:
            platt_coeffs = platt_scaling_fit(val_logits, y_val, mode="temp")
            val_preds = softmax(val_logits*platt_coeffs, axis=1)
            test_preds = softmax(test_logits*platt_coeffs, axis=1)
        else:
            val_preds = softmax(val_logits, axis=1)
            test_preds = softmax(test_logits, axis=1)
            
        qhat = LABEL_fit(y_val, val_preds, alpha=a, verbose=False)
        cs = LABEL_inference(test_preds,  qhat)
        
        cov.append(coverage_test(y_test, cs))
        bcov.append(balanced_coverage_test(y_test, cs))
        card.append(cardinality_test(cs))
    
    eff = [bcov[i]/card[i] for i in range(len(bcov))]
    return cov, bcov, card
    
# a method of evaluating the distribution of conformal prediction
# it performs X number of random splits to calibration and test sets
# and conformal prediction using each set
def develop_coverage_distribution(y, preds, n_splits=100, test_size=0.5, alpha=0.1):
    
    cov, bcov, card = [], [], []
    for i in range(n_splits):
        val_preds, test_preds, y_val, y_test = train_test_split(preds, y, test_size=test_size)
            
        qhat = class_balanced_LABEL_fit(y_val, val_preds, alpha=alpha, verbose=False)
        #print(qhat)
        cs = LABEL_inference(test_preds,  qhat)
        
        cov.append(coverage_test(y_test, cs))
        bcov.append(balanced_coverage_test(y_test, cs))
        card.append(cardinality_test(cs))
    
    eff = [bcov[i]/card[i] for i in range(len(bcov))]
    return cov, bcov, card