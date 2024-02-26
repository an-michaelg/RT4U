# Implementation of some conformal prediction algorithms
import numpy as np

def LABEL_fit(y, preds, alpha = 0.1, verbose=False):
    # y is (N,) integer array of C classes
    # preds is (N, C) array of confidences
    desired_accuracy = 1 - alpha
    N = len(y)
    pred_classes = np.argmax(preds,axis=1)
    pred_confidence = np.max(preds,axis=1)
    gt_confidence = preds[range(N),y]
    # calculate the conformal score s as a general imprecise uncertainty measure (can be many ways, this is a simple way)
    s = 1 - gt_confidence
    # find the q level
    q_level = np.ceil((N+1) * desired_accuracy)/N 
    qhat = np.quantile(s, q_level)
    if verbose:
        print(f"adjusted q-level {q_level}, q-hat {qhat}")
    return qhat


def LABEL_inference(preds, qhat):
    # preds is (N, C) array of confidences
    # qhat is computed from LABEL_fit
    pred_classes = np.argmax(preds,axis=1)
    cutoff = 1 - qhat
    N, C = preds.shape
    conformal_set = []
    for i in range(N):
        allowed = preds[i] >= cutoff
        if np.sum(allowed) == 0:
            #print(i)
            #print(preds[i])
            allowed[pred_classes[i]] = True
        conformal_set.append(allowed)
    return np.array(conformal_set)


# essentially running LABEL one time for each subgroup
def group_LABEL_fit(y, group_labels, preds, alpha=0.1, verbose=False):
    # y is (N,) integer array of C classes
    # group_labels is (N,) integer array of group labels (eg. different views)
    # preds is (N, C) array of confidences
    qhats = {}
    
    for subgroup in np.unique(group_labels):
        # find the conformal value
        y_subgroup = y[group_labels == subgroup]
        preds_subgroup = preds[group_labels == subgroup]
        if verbose:
            print(f"Group {subgroup}")
        qhats[subgroup] = LABEL_fit(y_subgroup, preds_subgroup, alpha, verbose)
        
    return qhats
        

def group_LABEL_inference(preds, group_labels, qhats):
    # group_labels is (N,) integer array of group labels (eg. different views)
    # preds is (N, C) array of confidences
    # qhats is a dictionary in the form of {subgroup_name:qhat}
    N, C = preds.shape
    conformal_set = []
    for i in range(N):
        subgroup = group_labels[i]
        subgroup_qhat = qhats[subgroup]
        conformal_set.append(preds[i] >= 1 - subgroup_qhat)
    return np.array(conformal_set)


def APS_fit(cal_labels, cal_smx, alpha=0.1, verbose=False):
    # Get scores. calib_X.shape[0] == calib_Y.shape[0] == n
    N = cal_smx.shape[0]
    
    # argsort returns the indices of the sort
    cal_pi = cal_smx.argsort(1)[:, ::-1]
    
    # take_along_axis returns results of the sort, then cumulative sum
    cal_srt = np.take_along_axis(cal_smx, cal_pi, axis=1).cumsum(axis=1)
    
    # this is some 5head stuff I dont get
    # but it returns "the cumulative sum by the time I get to class X"
    cal_scores = np.take_along_axis(cal_srt, cal_pi.argsort(axis=1), axis=1)[range(N), cal_labels]
    
    # Get the score quantile
    adj_q_level = np.ceil((N + 1) * (1 - alpha)) / N
    qhat = np.quantile(cal_scores, adj_q_level, interpolation="higher")
    if verbose:
        print(f"adjusted q-level {adj_q_level}, q-hat {qhat}")
    return qhat
    

def APS_inference(cal_smx, qhat):
    # Deploy (output=list of length n, each element is tensor of classes)
    cal_pi = cal_smx.argsort(1)[:, ::-1]
    
    cal_srt = np.take_along_axis(cal_smx, cal_pi, axis=1).cumsum(axis=1)
    
    prediction_sets = np.take_along_axis(cal_srt <= qhat, cal_pi.argsort(axis=1), axis=1)
    return prediction_sets


# essentially running APS one time for each subgroup
def group_APS_fit(y, group_labels, preds, alpha=0.1, verbose=False):
    # y is (N,) integer array of C classes
    # group_labels is (N,) integer array of group labels (eg. different views)
    # preds is (N, C) array of confidences
    qhats = {}
    
    for subgroup in np.unique(group_labels):
        # find the conformal value
        y_subgroup = y[group_labels == subgroup]
        preds_subgroup = preds[group_labels == subgroup]
        if verbose:
            print(f"Group {subgroup}")
        qhats[subgroup] = APS_fit(y_subgroup, preds_subgroup, alpha, verbose)
        
    return qhats
        

def group_APS_inference(preds, group_labels, qhats):
    # group_labels is (N,) integer array of group labels (eg. different views)
    # preds is (N, C) array of confidences
    # qhats is a dictionary in the form of {subgroup_name:qhat}
    N, C = preds.shape
    qhats_arr = []
    for i in range(N):
        qhats_arr.append(qhats[group_labels[i]])
    qhats_arr = np.expand_dims(np.array(qhats_arr), axis=1)
    assert qhats_arr.shape == (N, 1)
    
    return APS_inference(preds, qhats_arr)
    
    
def ordinal_APS_fit(y, preds, alpha = 0.1, verbose=False):
    return APS_fit(y, preds, alpha, verbose)
    

def ordinal_APS_inference(preds, qhat):
    N, C = preds.shape
    conformal_set = np.full(preds.shape, False)
    pred_class = np.argmax(preds, axis=1)
    for i in range(N):
        c = pred_class[i]
        q_init = preds[i, c]
        cmin = c
        cmax = c
        conformal_set[i, c] = True
        
        # grow the ordinal APS region until it exceeds qhat
        while q_init <= qhat:
            c_lower = (cmin - 1) if cmin > 0 else None
            p_lower = preds[i, c_lower] if c_lower is not None else -1.0
            c_higher = (cmax + 1) if cmax < C-1 else None
            p_higher = preds[i, c_higher] if c_higher is not None else -1.0
            #print(f"[{c_lower},{c_higher}]")
            #print(f"[{p_lower},{p_higher}]")
            # choose which new class to add
            if p_lower + p_higher == -2.0:
                # we exhausted the set, exit
                break
                
            if p_higher >= p_lower:
                cmax = c_higher
                q_init += preds[i, c_higher]
                conformal_set[i, c_higher] = True
            else: #p_lower > p_higher:
                cmin = c_lower
                q_init += preds[i, c_lower]
                conformal_set[i, c_lower] = True
            #print(conformal_set[i])
            #print(f"[{cmin},{cmax}]")
            #print("---")
            
    
    return conformal_set
    
def RAPS_fit(cal_smx, cal_labels, alpha=0.1, lam_reg=0.01, k_reg=5, verbose=False):
    # input hyperparameters
    n, c = cal_smx.shape
    # larger lam_reg and smaller k_reg leads to smaller sets
    assert k_reg < c
    disallow_zero_sets = False # Set this to False in order to see the coverage upper bound hold
    rand = True # Set this to True in order to see the coverage upper bound hold
    reg_vec = np.array(k_reg*[0,] + (smx.shape[1]-k_reg)*[lam_reg,])[None,:]
    
    # fit
    cal_pi = cal_smx.argsort(1)[:,::-1]; 
    cal_srt = np.take_along_axis(cal_smx,cal_pi,axis=1)
    cal_srt_reg = cal_srt + reg_vec
    cal_L = np.where(cal_pi == cal_labels[:,None])[1]
    cal_scores = cal_srt_reg.cumsum(axis=1)[np.arange(n),cal_L] - np.random.rand(n)*cal_srt_reg[np.arange(n),cal_L]
    # Get the score quantile
    adj_q_level = np.ceil((n+1)*(1-alpha))/n
    qhat = np.quantile(cal_scores, interpolation='higher')
    if verbose:
        print(f"adjusted q-level {adj_q_level}, q-hat {qhat}")
    return qhat
        
def RAPS_inference(val_smx, qhat, rand=True, disallow_zero_sets=False):
    # Set disallow to False in order to see the coverage upper bound hold
    # Set rand to True in order to see the coverage upper bound hold
    n_val = val_smx.shape[0]
    val_pi = val_smx.argsort(1)[:,::-1]
    val_srt = np.take_along_axis(val_smx,val_pi,axis=1)
    val_srt_reg = val_srt + reg_vec
    val_srt_reg_cumsum = val_srt_reg.cumsum(axis=1)
    indicators = (val_srt_reg.cumsum(axis=1) - np.random.rand(n_val,1)*val_srt_reg) <= qhat if rand else val_srt_reg.cumsum(axis=1) - val_srt_reg <= qhat
    if disallow_zero_sets: 
        indicators[:,0] = True
    prediction_sets = np.take_along_axis(indicators,val_pi.argsort(axis=1),axis=1)
    return prediction_sets