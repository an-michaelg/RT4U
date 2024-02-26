import numpy as np

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
    
# def cp_cardinality(y, preds_cover, preds, n_classes=4):
    # cardinality = np.sum(preds_cover, axis=1)
    # correct = coverage_test(y, preds_cover)
    # card_correct = cardinality[correct]
    # card_incorrect = cardinality[~correct]
    
    # n_bins = n_classes + 1
    # c_amount, c_bins = np.histogram(card_correct, bins=np.arange(n_bins)+0.5)
    # n_amount, _ = np.histogram(card_incorrect, bins=np.arange(n_bins)+0.5)
    # card_acc = c_amount / (c_amount + n_amount + 1e-9)
    # print(f"Number of correct preds of size [1..N_classes]: {c_amount}")
    # print(f"Number of incorrect preds of size [1..N_classes]: {n_amount}")
    # print(f"Accuracy of preds of size [1..N_classes]: {card_acc}")

    # top1_accs = []
    # pred_classes = np.argmax(preds, axis=1)
    # for i in np.arange(n_classes)+1:
        # gt_w_card_i = y[cardinality==i]
        # pred_w_card_i = pred_classes[cardinality==i]
        # acc_w_card_i = np.sum(gt_w_card_i == pred_w_card_i) / (len(gt_w_card_i) + 1e-9)
        # top1_accs.append(acc_w_card_i)
    # print(f"Top-1 acc of preds of size [1..N_classes]: {top1_accs}")
    # return np.mean(cardinality)

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