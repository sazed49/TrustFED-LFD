import copy
import enum
import torch
import numpy as np
import math
from scipy import stats
from functools import reduce
import time
import sklearn.metrics.pairwise as smp
import hdbscan
from scipy.spatial.distance import cdist
from scipy.stats import entropy
from sklearn.cluster import KMeans
from utils import *
from sklearn.metrics import silhouette_score



def get_pca(data, threshold = 0.99):
    normalized_data = StandardScaler().fit_transform(data)
    pca = PCA()
    reduced_data = pca.fit_transform(normalized_data)
    # Determine explained variance using explained_variance_ration_ attribute
    exp_var = pca.explained_variance_ratio_
    cum_sum_eigenvalues = np.cumsum(exp_var)
    select_pcas = np.where(cum_sum_eigenvalues <=threshold)[0]
    # print('Number of components with variance <= {:0.0f}%: {}'.format(threshold*100, len(select_pcas)))
    reduced_data = reduced_data[:, select_pcas]
    return reduced_data

eps = np.finfo(float).eps




class RobustLFD_Optimized:
    """
    Faster RobustLFD:
      - No sklearn KMeans
      - No silhouette_score
      - No huge flatten for clustering
      - Uses small feature vector (4 dims) from top-2 suspicious classes
      - Uses cheap 1D projection + median split for 2 clusters
      - Vectorized weighted averaging (torch stack) to reduce Python overhead
      - Uses robust-ish outlier rejection (median + MAD) instead of mean/std
    """

    def __init__(
        self,
        num_classes: int,
        rejection_k: float = 3.5,      # how strict outlier rejection is (MAD scale)
        memory_decay: float = 0.9,     # decay for memory accumulation
        eps: float = 1e-12,
        device: str = "cpu",
    ):
        self.memory = np.zeros([num_classes], dtype=np.float64)
        self.rejection_k = rejection_k
        self.memory_decay = memory_decay
        self.eps = eps
        self.device = device

    @staticmethod
    def _flatten_last_layer(W: torch.Tensor) -> torch.Tensor:
        # W is typically [num_classes, hidden_dim] (FC weight)
        return W.reshape(W.shape[0], -1)

    def _robust_reject(self, dW: torch.Tensor) -> torch.Tensor:
        """
        dW: [m, C, H] (or [m, C, ...] after reshape)
        Return: mask of valid clients (bool tensor of shape [m])
        """
        m = dW.shape[0]
        flat = dW.reshape(m, -1)
        norms = torch.norm(flat, dim=1)  # [m]

        med = torch.median(norms)
        mad = torch.median(torch.abs(norms - med)) + self.eps
        thr = med + self.rejection_k * mad

        valid = norms <= thr
        return valid

    def _select_top2_classes(self, dW: torch.Tensor, db: torch.Tensor) -> np.ndarray:
        """
        dW: [m, C, H]
        db: [m, C]
        Update memory and return indices of top-2 suspicious classes.
        """
        # per-class magnitude from weights + bias
        # weights magnitude per class: sum over clients and hidden dims
        dw_per_class = torch.sum(torch.abs(dW), dim=(0, 2)).detach().cpu().numpy()  # [C]
        db_per_class = torch.sum(torch.abs(db), dim=0).detach().cpu().numpy()       # [C]

        # decay + accumulate
        self.memory = self.memory_decay * self.memory + (dw_per_class + db_per_class)

        top2 = np.argsort(self.memory)[-2:]  # 2 classes
        return top2

    def _build_small_features(self, dW: torch.Tensor, top2: np.ndarray) -> torch.Tensor:
        """
        Build tiny feature vector per client using only 2 class-rows.
        dW: [m, C, H]
        top2: array([c1, c2])
        Return X: [m, 4] on CPU (float32)
        Features:
          - L2 norm of row c1
          - L2 norm of row c2
          - mean abs of row c1
          - mean abs of row c2
        """
        rows = dW[:, top2, :]  # [m, 2, H]
        l2 = torch.norm(rows, dim=2)                 # [m, 2]
        mean_abs = torch.mean(torch.abs(rows), dim=2)  # [m, 2]
        X = torch.cat([l2, mean_abs], dim=1)         # [m, 4]
        return X.to(torch.float32).cpu()

    def _cheap_two_cluster_split(self, X: torch.Tensor) -> torch.Tensor:
        """
        X: [m, d] (CPU)
        Return labels: [m] in {0,1} using projection+median split.
        """
        # direction = mean feature vector
        u = torch.mean(X, dim=0)  # [d]
        u_norm = torch.norm(u) + self.eps
        u = u / u_norm

        proj = X @ u  # [m]
        thr = torch.median(proj)
        labels = (proj > thr).to(torch.int64)
        return labels

    def _soft_scores_by_centroid(self, X: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        X: [m, d] CPU
        labels: [m] CPU {0,1}
        Return scores: [m] CPU float32, exp(-dist_to_own_centroid)
        """
        m = X.shape[0]
        scores = torch.ones(m, dtype=torch.float32)

        idx0 = (labels == 0).nonzero(as_tuple=False).squeeze(-1)
        idx1 = (labels == 1).nonzero(as_tuple=False).squeeze(-1)

        # If a cluster is too small, fall back to uniform weights
        if idx0.numel() < 2 or idx1.numel() < 2:
            return scores

        c0 = torch.mean(X[idx0], dim=0)
        c1 = torch.mean(X[idx1], dim=0)

        # distances to own centroid
        d0 = torch.norm(X[idx0] - c0, dim=1)
        d1 = torch.norm(X[idx1] - c1, dim=1)

        scores[idx0] = torch.exp(-d0)
        scores[idx1] = torch.exp(-d1)

        # avoid all-zero / tiny sum
        if torch.sum(scores) < self.eps:
            scores = torch.ones_like(scores)

        return scores

    def _weighted_average_state_dicts(self, local_weights, scores: torch.Tensor):
        """
        local_weights: list of state_dict
        scores: [m] CPU torch tensor
        Return averaged state_dict
        """
        m = len(local_weights)
        scores = scores.to(torch.float32)
        total = torch.sum(scores) + self.eps

        averaged = copy.deepcopy(local_weights[0])
        # We will compute on CPU to avoid device mismatches in state_dict
        for k in averaged.keys():
            # Stack [m, ...]
            stacked = torch.stack([local_weights[i][k].detach().cpu() for i in range(m)], dim=0)
            # Broadcast scores: [m, 1, 1, ...]
            view_shape = [m] + [1] * (stacked.dim() - 1)
            wsum = torch.sum(stacked * scores.view(*view_shape), dim=0) / total
            averaged[k] = wsum
        return averaged

    def aggr(self, global_model, local_models, ptypes=None):
        """
        Keeps same signature style as your RobustLFD.aggr.
        Returns: new global weights (state_dict)
        """
        print("inside robust aggregate (optimized)")

        # snapshot local weights for averaging later
        local_weights = [copy.deepcopy(m).state_dict() for m in local_models]
        m = len(local_models)
        if m == 0:
            raise ValueError("No local models provided.")

        # Extract last layer params
        # (Assumes last two parameters are FC weight and bias)
        g_params = list(global_model.parameters())
        gW = g_params[-2].detach()
        gb = g_params[-1].detach()

        # Build dW, db in torch
        # Move to target device for fast math (default CPU; you can set "cuda" if available)
        gW_t = gW.to(self.device)
        gb_t = gb.to(self.device)

        dW_list = []
        db_list = []
        for lm in local_models:
            lp = list(lm.parameters())
            lW = lp[-2].detach().to(self.device)
            lb = lp[-1].detach().to(self.device)
            dW_list.append(gW_t - lW)
            db_list.append(gb_t - lb)

        dW = torch.stack(dW_list, dim=0)  # [m, C, H]
        db = torch.stack(db_list, dim=0)  # [m, C]

        # 1) reject extreme updates (robust)
        valid_mask = self._robust_reject(dW)  # [m] bool
        valid_idx = valid_mask.nonzero(as_tuple=False).squeeze(-1).detach().cpu().numpy()

        if len(valid_idx) < m:
            print(f"Rejected {m - len(valid_idx)} clients for extreme gradients (MAD).")

        # If too few remain, fall back to simple average
        if len(valid_idx) < 2:
            print("Too few valid clients. Using all clients with uniform weights.")
            scores = torch.ones(m, dtype=torch.float32)
            return self._weighted_average_state_dicts(local_weights, scores)

        # filter tensors and weights
        dW_v = dW[valid_mask]  # [mv, C, H]
        db_v = db[valid_mask]  # [mv, C]
        local_weights_v = [local_weights[i] for i in valid_idx.tolist()] if hasattr(valid_idx, "tolist") else [local_weights[i] for i in valid_idx]

        mv = dW_v.shape[0]

        # 2) choose top2 classes (only if multi-class)
        if db_v.shape[1] > 2:
            top2 = self._select_top2_classes(dW_v, db_v)
            X = self._build_small_features(dW_v, top2)  # [mv,4]
        else:
            # binary: still build small features from both classes if possible
            top2 = np.array([0, 1]) if db_v.shape[1] == 2 else np.array([0, 0])
            if db_v.shape[1] >= 2:
                X = self._build_small_features(dW_v, top2)  # [mv,4]
            else:
                # fallback: norm + meanabs of entire dW
                flat = dW_v.reshape(mv, -1).detach().cpu()
                X = torch.stack([torch.norm(flat, dim=1), torch.mean(torch.abs(flat), dim=1),
                                 torch.norm(flat, dim=1), torch.mean(torch.abs(flat), dim=1)], dim=1)

        # 3) cheap 2-cluster split (no KMeans)
        labels = self._cheap_two_cluster_split(X)  # [mv]

        # 4) soft scores within clusters
        scores_v = self._soft_scores_by_centroid(X, labels)  # [mv]

        # 5) weighted average only over valid clients
        new_global = self._weighted_average_state_dicts(local_weights_v, scores_v)
        return new_global



class RobustLFD:
    def __init__(self, num_classes, rejection_threshold=2.5):
        self.memory = np.zeros([num_classes])
        self.rejection_threshold = rejection_threshold  # For gradient norm outlier rejection

    def score_clients(self, data, labels):
        """Soft scoring using distance to cluster centroid."""
        scores = np.ones(len(labels))
        clusters = {0: [], 1: []}
        for i, l in enumerate(labels):
            clusters[l].append(data[i])
        centroids = [np.mean(clusters[0], axis=0), np.mean(clusters[1], axis=0)]
        for i, l in enumerate(labels):
            dist = np.linalg.norm(data[i] - centroids[l])
            # Inverse distance scoring; closer = more trustworthy
            scores[i] = np.exp(-dist)
        return scores

    def aggr(self, global_model, local_models, ptypes):
        print("inside robust aggregate")
        local_weights = [copy.deepcopy(model).state_dict() for model in local_models]
        m = len(local_models)

        # Extract weight differences from last FC layer
        for i in range(m):
            local_models[i] = list(local_models[i].parameters())
        global_model = list(global_model.parameters())

        dw = np.array([global_model[-2].cpu().data.numpy() - local_models[i][-2].cpu().data.numpy() for i in range(m)])
        db = np.array([global_model[-1].cpu().data.numpy() - local_models[i][-1].cpu().data.numpy() for i in range(m)])
        
        # Reject extreme gradient clients (gradient magnitude outliers)
        norms = np.linalg.norm(dw, axis=1)
        threshold = np.mean(norms) + self.rejection_threshold * np.std(norms)
        valid_indices = np.where(norms < threshold)[0]
        if len(valid_indices) < m:
            print(f"Rejected {m - len(valid_indices)} clients for extreme gradients.")
        dw = dw[valid_indices]
        db = db[valid_indices]
        local_weights = [local_weights[i] for i in valid_indices]

        # Use only top-2 most flipped classes (if multi-class)
        if db.shape[1] > 2:
            # Assuming the output layer has shape [10] and corresponds to classes
            # Extract the row that corresponds to output layer (dw has shape [num_clients, output_dim])
            dw_per_class = np.sum(np.abs(dw), axis=(0, 2))
            self.memory += dw_per_class

            
            
            #self.memory += np.sum(np.abs(dw), axis=0)
            top2 = self.memory.argsort()[-2:]
            data = dw[:, top2].reshape(len(dw), -1)
        else:
            data = dw.reshape(len(dw), -1)

        # Clustering
        kmeans = KMeans(n_clusters=2, random_state=0).fit(data)
        labels = kmeans.labels_

        # Validate clustering
        if silhouette_score(data, labels) < 0.1:
            print("Clustering failed (low silhouette). Using all clients.")
            scores = np.ones(len(data))
        else:
            scores = self.score_clients(data, labels)

        global_weights = self.average_weights(local_weights, scores)
        return global_weights

    def average_weights(self, local_weights, scores):
        """Weighted average of models."""
        total_score = np.sum(scores)
        averaged = copy.deepcopy(local_weights[0])
        for key in averaged.keys():
            averaged[key] = sum(w[key] * s for w, s in zip(local_weights, scores)) / total_score
        return averaged

class LFD():
    def __init__(self, num_classes):
        self.memory = np.zeros([num_classes])
    
    def clusters_dissimilarity(self, clusters):
        n0 = len(clusters[0])
        n1 = len(clusters[1])
        m = n0 + n1 
        cs0 = smp.cosine_similarity(clusters[0]) - np.eye(n0)
        cs1 = smp.cosine_similarity(clusters[1]) - np.eye(n1)
        mincs0 = np.min(cs0, axis=1)
        mincs1 = np.min(cs1, axis=1)
        ds0 = n0/m * (1 - np.mean(mincs0))
        ds1 = n1/m * (1 - np.mean(mincs1))
        return ds0, ds1

    def aggregate(self, global_model, local_models, ptypes):
        print("aggregate a achi")
        local_weights = [copy.deepcopy(model).state_dict() for model in local_models]
        m = len(local_models)
        for i in range(m):
            local_models[i] = list(local_models[i].parameters())
        global_model = list(global_model.parameters())
        dw = [None for i in range(m)]
        db = [None for i in range(m)]
        for i in range(m):
            dw[i]= global_model[-2].cpu().data.numpy() - \
                local_models[i][-2].cpu().data.numpy() 
            db[i]= global_model[-1].cpu().data.numpy() - \
                local_models[i][-1].cpu().data.numpy()
        dw = np.asarray(dw)
        db = np.asarray(db)

        "If one class or two classes classification model"
        if len(db[0]) <= 2:
            data = []
            for i in range(m):
                data.append(dw[i].reshape(-1))
        
            kmeans = KMeans(n_clusters=2, random_state=0).fit(data)
            labels = kmeans.labels_

            clusters = {0:[], 1:[]}
            for i, l in enumerate(labels):
                clusters[l].append(data[i])

            good_cl = 0
            cs0, cs1 = self.clusters_dissimilarity(clusters)
            if cs0 < cs1:
                good_cl = 1

            # print('Cluster 0 weighted variance', cs0)
            # print('Cluster 1 weighted variance', cs1)
            # print('Potential good cluster is:', good_cl)
            scores = np.ones([m])
            malicious_clients=[]
            for i, l in enumerate(labels):
                # print(ptypes[i], 'Cluster:', l)
                if l != good_cl:
                    scores[i] = 0
                    malicious_clients.append(i)
            print("Detected Attackers->",malicious_clients)    
            global_weights = average_weights(local_weights, scores)
            return global_weights

        "For multiclassification models"
        norms = np.linalg.norm(dw, axis = -1) 
        self.memory = np.sum(norms, axis = 0)
        self.memory +=np.sum(abs(db), axis = 0)
        max_two_freq_classes = self.memory.argsort()[-2:]
        print('Potential source and target classes:', max_two_freq_classes)
        data = []
        for i in range(m):
            data.append(dw[i][max_two_freq_classes].reshape(-1))

        kmeans = KMeans(n_clusters=2, random_state=0).fit(data)
        labels = kmeans.labels_

        clusters = {0:[], 1:[]}
        for i, l in enumerate(labels):
          clusters[l].append(data[i])

        good_cl = 0
        cs0, cs1 = self.clusters_dissimilarity(clusters)
        if cs0 < cs1:
            good_cl = 1

        # print('Cluster 0 weighted variance', cs0)
        # print('Cluster 1 weighted variance', cs1)
        # print('Potential good cluster is:', good_cl)
        scores = np.ones([m])
        for i, l in enumerate(labels):
            # print(ptypes[i], 'Cluster:', l)
            if l != good_cl:
                scores[i] = 0
            
        global_weights = average_weights(local_weights, scores)
        return global_weights

################################################
# Takes in grad
# Compute similarity
# Get weightings
def foolsgold(grads):
    n_clients = grads.shape[0]
    cs = smp.cosine_similarity(grads) - np.eye(n_clients)
    maxcs = np.max(cs, axis=1)
    # pardoning
    for i in range(n_clients):
        for j in range(n_clients):
            if i == j:
                continue
            if maxcs[i] < maxcs[j]:
                cs[i][j] = cs[i][j] * maxcs[i] / maxcs[j]
    wv = 1 - (np.max(cs, axis=1))
    wv[wv > 1] = 1
    wv[wv < 0] = 0

    # Rescale so that max value is wv
    wv = wv / np.max(wv)
    wv[(wv == 1)] = .99

    # Logit function
    wv = (np.log(wv / (1 - wv)) + 0.5)
    wv[(np.isinf(wv) + wv > 1)] = 1
    wv[(wv < 0)] = 0

    return wv


class FoolsGold:
    def __init__(self, num_peers):
        self.memory = None
        self.wv_history = []
        self.num_peers = num_peers
       
    def score_gradients(self, local_grads, selectec_peers):
        m = len(local_grads)
        grad_len = np.array(local_grads[0][-2].cpu().data.numpy().shape).prod()
        if self.memory is None:
            self.memory = np.zeros((self.num_peers, grad_len))

        grads = np.zeros((m, grad_len))
        for i in range(m):
            grads[i] = np.reshape(local_grads[i][-2].cpu().data.numpy(), (grad_len))
        self.memory[selectec_peers]+= grads
        wv = foolsgold(self.memory)  # Use FG
        self.wv_history.append(wv)
        return wv[selectec_peers]


#######################################################################################
class Tolpegin:
    def __init__(self):
        pass
    
    def score(self, global_model, local_models, peers_types, selected_peers):
        global_model = list(global_model.parameters())
        last_g = global_model[-2].cpu().data.numpy()
        m = len(local_models)
        grads = [None for i in range(m)]
        for i in range(m):
            grad= (last_g - \
                    list(local_models[i].parameters())[-2].cpu().data.numpy())
            grads[i] = grad
        
        grads = np.array(grads)
        num_classes = grad.shape[0]
        # print('Number of classes:', num_classes)
        dist = [ ]
        labels = [ ]
        for c in range(num_classes):
            data = grads[:, c]
            data = get_pca(copy.deepcopy(data))
            kmeans = KMeans(n_clusters=2, random_state=0).fit(data)
            cl = kmeans.cluster_centers_
            dist.append(((cl[0] - cl[1])**2).sum())
            labels.append(kmeans.labels_)
        
        dist = np.array(dist)
        candidate_class = dist.argmax()
        print("Candidate source/target class", candidate_class)
        labels = labels[candidate_class]
        if sum(labels) < m/2:
            scores = 1 - labels
        else:
            scores = labels
        
        for i, pt in enumerate(peers_types):
            print(pt, 'scored', scores[i])
        return scores
#################################################################################################################
# Clip local updates
def clipp_model(g_w, w, gamma =  1):
    for layer in w.keys():
        w[layer] = g_w[layer] + (w[layer] - g_w[layer])*min(1, gamma)
    return w
def FLAME(global_model, local_models, noise_scalar):
    # Compute number of local models
    m = len(local_models)
    
    # Flattent local models
    g_m = np.array([torch.nn.utils.parameters_to_vector(global_model.parameters()).cpu().data.numpy()])
    f_m = np.array([torch.nn.utils.parameters_to_vector(model.parameters()).cpu().data.numpy() for model in local_models])
    grads = g_m - f_m
    # Compute model-wise cosine similarity
    cs = smp.cosine_similarity(grads)
    # Compute the minimum cluster size value
    msc = int(m*0.5) + 1 
    # Apply HDBSCAN on the computed cosine similarities
    clusterer = hdbscan.HDBSCAN(min_cluster_size=msc, min_samples=1, allow_single_cluster = True)
    clusterer.fit(cs)
    labels = clusterer.labels_
    # print('Clusters:', labels)

    if sum(labels) == -(m):
        # In case all of the local models identified as outliers, consider all of as benign
        benign_idxs = np.arange(m)
    else:
        benign_idxs = np.where(labels!=-1)[0]
        
    # Compute euclidean distances to the current global model
    euc_d = cdist(g_m, f_m)[0]
    # Identify the median of computed distances
    st = np.median(euc_d)
    # Clipp admitted updates
    W_c = []
    for i, idx in enumerate(benign_idxs):
        w_c = clipp_model(global_model.state_dict(), local_models[idx].state_dict(), gamma =  st/euc_d[idx])
        W_c.append(w_c)
    
    # Average admitted clipped updates to obtain a new global model
    g_w = average_weights(W_c, np.ones(len(W_c)))
    
    '''From the original paper: {We use standard DP parameters and set eps = 3705 for IC, 
    eps = 395 for the NIDS and eps = 4191 for the NLP scenario. 
    Accordingly, lambda = 0.001 for IC and NLP, and lambda = 0.01 for the NIDS scenario.}
    However, we found lambda = 0.001 with the CIFAR10-ResNet18 benchmark spoils the model
    and therefore we tried lower lambda values, which correspond to greater eps values.'''
    
    # Add adaptive noise to the global model
    lamb = 0.001
    sigma = lamb*st*noise_scalar
    # print('Sigma:{:0.4f}'.format(sigma))
    for key in g_w.keys():
        noise = torch.FloatTensor(g_w[key].shape).normal_(mean=0, std=(sigma**2)).to(g_w[key].device)
        g_w[key] = g_w[key] + noise
        
    return g_w 
#################################################################################################################

def median_opt(input):
    shape = input.shape
    input = input.sort()[0]
    if shape[-1] % 2 != 0:
        output = input[..., int((shape[-1] - 1) / 2)]
    else:
        output = (input[..., int(shape[-1] / 2 - 1)] + input[..., int(shape[-1] / 2)]) / 2.0
    return output

def Repeated_Median_Shard(w):
    SHARD_SIZE = 100000
    w_med = copy.deepcopy(w[0])
    device = w[0][list(w[0].keys())[0]].device

    for k in w_med.keys():
        shape = w_med[k].shape
        if len(shape) == 0:
            continue
        total_num = reduce(lambda x, y: x * y, shape)
        y_list = torch.FloatTensor(len(w), total_num).to(device)
        for i in range(len(w)):
            y_list[i] = torch.reshape(w[i][k], (-1,))
        y = torch.t(y_list)

        if total_num < SHARD_SIZE:
            slopes, intercepts = repeated_median(y)
            y = intercepts + slopes * (len(w) - 1) / 2.0
        else:
            y_result = torch.FloatTensor(total_num).to(device)
            assert total_num == y.shape[0]
            num_shards = int(math.ceil(total_num / SHARD_SIZE))
            for i in range(num_shards):
                y_shard = y[i * SHARD_SIZE: (i + 1) * SHARD_SIZE, ...]
                slopes_shard, intercepts_shard = repeated_median(y_shard)
                y_shard = intercepts_shard + slopes_shard * (len(w) - 1) / 2.0
                y_result[i * SHARD_SIZE: (i + 1) * SHARD_SIZE] = y_shard
            y = y_result
        y = y.reshape(shape)
        w_med[k] = y
    return w_med


def repeated_median(y):
    num_models = y.shape[1]
    total_num = y.shape[0]
    y = y.sort()[0]
    yyj = y.repeat(1, 1, num_models).reshape(total_num, num_models, num_models)
    yyi = yyj.transpose(-1, -2)
    xx = torch.FloatTensor(range(num_models)).to(y.device)
    xxj = xx.repeat(total_num, num_models, 1)
    xxi = xxj.transpose(-1, -2) + eps

    diag = torch.Tensor([float('Inf')] * num_models).to(y.device)
    diag = torch.diag(diag).repeat(total_num, 1, 1)

    dividor = xxi - xxj + diag
    slopes = (yyi - yyj) / dividor + diag
    slopes, _ = slopes.sort()
    slopes = median_opt(slopes[:, :, :-1])
    slopes = median_opt(slopes)

    # get intercepts (intercept of median)
    yy_median = median_opt(y)
    xx_median = [(num_models - 1) / 2.0] * total_num
    xx_median = torch.Tensor(xx_median).to(y.device)
    intercepts = yy_median - slopes * xx_median

    return slopes, intercepts


# Repeated Median estimator
def Repeated_Median(w):
    cur_time = time.time()
    w_med = copy.deepcopy(w[0])
    device = w[0][list(w[0].keys())[0]].device

    for k in w_med.keys():
        shape = w_med[k].shape
        if len(shape) == 0:
            continue
        total_num = reduce(lambda x, y: x * y, shape)
        y_list = torch.FloatTensor(len(w), total_num).to(device)
        for i in range(len(w)):
            y_list[i] = torch.reshape(w[i][k], (-1,))
        y = torch.t(y_list)

        slopes, intercepts = repeated_median(y)
        y = intercepts + slopes * (len(w) - 1) / 2.0

        y = y.reshape(shape)
        w_med[k] = y

    print('repeated median aggregation took {}s'.format(time.time() - cur_time))
    return w_med
        
# simple median estimator
def simple_median(w):
    device = w[0][list(w[0].keys())[0]].device
    w_med = copy.deepcopy(w[0])
    for k in w_med.keys():
        shape = w_med[k].shape
        if len(shape) == 0:
            continue
        total_num = reduce(lambda x, y: x * y, shape)
        y_list = torch.FloatTensor(len(w), total_num).to(device)
        for i in range(len(w)):
            y_list[i] = torch.reshape(w[i][k], (-1,))
        y = torch.t(y_list)
        median_result = median_opt(y)
        assert total_num == len(median_result)

        weight = torch.reshape(median_result, shape)
        w_med[k] = weight
    return w_med

def trimmed_mean(w, trim_ratio):
    if trim_ratio == 0:
        return average_weights(w, [1 for i in range(len(w))])
        
    assert trim_ratio < 0.5, 'trim ratio is {}, but it should be less than 0.5'.format(trim_ratio)
    trim_num = int(trim_ratio * len(w))
    device = w[0][list(w[0].keys())[0]].device
    w_med = copy.deepcopy(w[0])
    for k in w_med.keys():
        shape = w_med[k].shape
        if len(shape) == 0:
            continue
        total_num = reduce(lambda x, y: x * y, shape)
        y_list = torch.FloatTensor(len(w), total_num).to(device)
        for i in range(len(w)):
            y_list[i] = torch.reshape(w[i][k], (-1,))
        y = torch.t(y_list)
        y_sorted = y.sort()[0]
        result = y_sorted[:, trim_num:-trim_num]
        result = result.mean(dim=-1)
        assert total_num == len(result)

        weight = torch.reshape(result, shape)
        w_med[k] = weight
    return w_med


# Get average weights
def average_weights(w, marks):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        w_avg[key] = w_avg[key] * marks[0]
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key] * marks[i]
        w_avg[key] = w_avg[key] *(1/sum(marks))
    return w_avg
   
def Krum(updates, f, multi = False):
    n = len(updates)
    updates = [torch.nn.utils.parameters_to_vector(update.parameters()) for update in updates]
    updates_ = torch.empty([n, len(updates[0])])
    for i in range(n):
      updates_[i] = updates[i]
    k = n - f - 2
    # collection distance, distance from points to points
    cdist = torch.cdist(updates_, updates_, p=2)
    dist, idxs = torch.topk(cdist, k , largest=False)
    dist = dist.sum(1)
    idxs = dist.argsort()
    if multi:
      return idxs[:k]
    else:
      return idxs[0]
##################################################################
