import numpy as np


def calculate_conf_matrix(pred, label):
    TP = np.sum(label[pred == 1] == pred[pred == 1])
    FP = np.sum(label[pred == 1] != pred[pred == 1])
    FN = np.sum(label[pred == 0] != pred[pred == 0])
    TN = np.sum(label[pred == 0] == pred[pred == 0])
    return TP, FP, FN, TN


def calculate_index(pred, label, index_name=None):
    if index_name is None:
        index_name = np.array(["accuracy", "error rate", "precision", "negative precision", 
                               "sensitivity", "specificity", "false positive rate", "false negative rate"])
    else:
        index_name = np.array(index_name)
        
    permutation = []
    indices = []
    
    if ("accuracy" in index_name):
        accuracy = (TP + TN) / (TP + FP + FN + TN)
        indices.append(accuracy)
        permutation.append(np.where("accuracy" == index_name)[0][0])
            
    if ("error rate" in index_name):
        error_rate = 1 - accuracy
        indices.append(error_rate)
        permutation.append(np.where("error rate" == index_name)[0][0])
    
    if ("precision" in index_name):
        precision = TP / (TP + FP)
        indices.append(precision)
        permutation.append(np.where("precision" == index_name)[0][0])
        
    if ("negative precision" in index_name):
        nega_precision = TN / (TN + FN)
        indices.append(nega_precision)
        permutation.append(np.where("negative precision" == index_name)[0][0])
    
    if ("sensitivity" in index_name):
        sensitivity = TP / (TP + FN)
        indices.append(sensitivity)
        permutation.append(np.where("sensitivity" == index_name)[0][0])
    
    if ("specificity" in index_name):
        specificity = TN / (TN + FP)
        indices.append(specificity)
        permutation.append(np.where("specificity" == index_name)[0][0])
    
    if ("false positive rate" in index_name):
        false_pos_rate = FP / (TN + FP)
        indices.append(false_pos_rate)
        permutation.append(np.where("false positive rate" == index_name)[0][0])
    
    if ("false negative rate" in index_name):
        false_neg_rate = FN / (TP + FN)
        indices.append(false_neg_rate)
        permutation.append(np.where("false negative rate" == index_name)[0][0])
    
    indices = np.array(indices)
    permutation = np.array(permutation)
    
    if index_name is not None:
        indices[permutation]
        
    return indices