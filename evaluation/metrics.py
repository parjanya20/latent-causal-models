from itertools import permutations
import numpy as np

def get_shd_f1(matrix1, matrix2, n_x):
    '''
    Input: ground truth matrix, predicted matrix, number of observed variables
    Output: SHD, F1 score, permutation of the predicted matrix that matches the ground truth matrix
    '''
    if not isinstance(matrix2, np.ndarray):
        matrix2 = matrix2.cpu().detach().numpy()
    matrix2 = (matrix2 > 0.5).astype(int)
    
    def preprocess_matrix(matrix):
        n = matrix.shape[0]
        indices_to_keep = []
        for i in range(n):
            if np.any(matrix[i, :] != 0) or np.any(matrix[:, i] != 0):
                indices_to_keep.append(i)
        return matrix[indices_to_keep][:, indices_to_keep]
    
    def calculate_shd(m1, m2):
        return np.sum(m1 != m2)
    
    def calculate_f1(m1, m2):
        true_positives = np.sum((m1 == 1) & (m2 == 1))
        false_positives = np.sum((m1 == 0) & (m2 == 1))
        false_negatives = np.sum((m1 == 1) & (m2 == 0))
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        return f1
    
    matrix1 = preprocess_matrix(matrix1)
    matrix2 = preprocess_matrix(matrix2)
    
    if matrix1.shape[0] < matrix2.shape[0]:
        diff = matrix2.shape[0] - matrix1.shape[0]
        matrix1 = np.pad(matrix1, ((diff, 0), (diff, 0)), mode='constant')
    elif matrix2.shape[0] < matrix1.shape[0]:
        diff = matrix1.shape[0] - matrix2.shape[0]
        matrix2 = np.pad(matrix2, ((diff, 0), (diff, 0)), mode='constant')
    
    n = matrix1.shape[0]
    n_z = n - n_x
    
    if n_z > n:
        raise ValueError("n_z is larger than the matrix size after preprocessing")
    
    perms = list(permutations(range(n_z)))
    min_shd = float('inf')
    max_f1 = -float('inf')
    best_perm = None
    
    for perm1 in perms:
        for perm in perms:
            full_perm = list(perm) + list(range(n_z, n))
            full_perm1 = list(perm1) + list(range(n_z, n))
            
            permuted_matrix2 = matrix2[full_perm][:, full_perm]
            permuted_matrix1 = matrix1[full_perm1][:, full_perm1]
            
            current_shd = calculate_shd(permuted_matrix1, permuted_matrix2)
            current_f1 = calculate_f1(permuted_matrix1, permuted_matrix2)
            
            if current_shd < min_shd:
                min_shd = current_shd
                max_f1 = current_f1
                best_perm = perm
            elif current_shd == min_shd and current_f1 > max_f1:
                max_f1 = current_f1
                best_perm = perm
    
    return min_shd, max_f1, best_perm
