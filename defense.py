#import hdbscan
#from sklearn.metrics.pairwise import pairwise_distances
import numpy as np
#from scipy.special import logit
import sklearn.metrics.pairwise as smp
from helpers.func_utils import compute_euclidean_distance


def fedavg(args, grad_in):
    grad = np.array(grad_in).reshape((args.num_workers, -1)).mean(axis=0)
    return grad.tolist()

def foolsgold(args, grad_history, grad_in):
    epsilon = 1e-5
    grad_in = np.array(grad_in).reshape((args.num_workers, -1))
    grad_history = np.array(grad_history)
    if grad_history.shape[0] != args.num_workers:
        grad_history = grad_history[:args.num_workers,:] + grad_history[args.num_workers:,:]

    similarity_maxtrix = smp.cosine_similarity(grad_history) - np.eye(args.num_workers)

    mv = np.max(similarity_maxtrix, axis=1) + epsilon

    alpha = np.zeros(mv.shape)
    for i in range(args.num_workers):
        for j in range(args.num_workers):
            if mv[j] > mv[i]:
                similarity_maxtrix[i,j] *= mv[i]/mv[j]

    alpha = 1 - (np.max(similarity_maxtrix, axis=1))
    alpha[alpha > 1] = 1
    alpha[alpha < 0] = 0
    alpha = alpha/np.max(alpha)
    alpha[(alpha == 1)] = 0.99
    alpha = (np.log((alpha / (1 - alpha)) + epsilon) + 0.5)
    alpha[(np.isinf(alpha) + alpha > 1)] = 1
    alpha[(alpha < 0)] = 0
    print("alpha:")
    print(alpha)
    grad = np.average(grad_in, weights=alpha, axis=0)
    return grad.tolist(), grad_history.tolist(), alpha

# attack setting i.e., 1%, 2%, and 5%
# To do add the defense method
# 1.Median
# 2.Trimmed-mean
# 5.Norm-bounding


# 3.Krum
# 4.MultiKrum
def _compute_krum_score(args, vec_grad_list):
    krum_scores = []
    num_client = len(vec_grad_list)
    for i in range(0, num_client):
        dists = []
        for j in range(0, num_client):
            if i != j:
                dists.append(
                    compute_euclidean_distance(
                        vec_grad_list[i], vec_grad_list[j]
                    ).item() ** 2
                )
        dists.sort()  # ascending
        score = dists[0 : num_client - args.num_mali - 2]
        krum_scores.append(sum(score))
    return krum_scores

def MultiKrum(args,grad_in):
    """
    :param args: paramaters, should include num_mali: byzantine_client_num, krum_param_m
    :param grad_in: model weights list. grad_in[i] i-th clients's model wiights
    :return: aggregation grad_in, should keep the same shape with original grad_in
    """
    grad_in = np.array(grad_in).reshape((args.num_workers, -1))
    num_client = len(grad_in)
    # in the Krum paper, it says 2 * byzantine_client_num + 2 < client #
    if not 2 * args.num_mali + 2 <= num_client - args.krum_param_m:
        raise ValueError(
            "byzantine_client_num conflicts with requirements in Krum: 2 * byzantine_client_num + 2 < client number - krum_param_m"
        )

    vec_local_w = [
        grad_in[i]
        for i in range(0, num_client)
    ]
    krum_scores = _compute_krum_score(vec_local_w)

    score_index = np.array(krum_scores).argsort().tolist()
    krum_grad_in = grad_in[score_index]
    grad = np.average(krum_grad_in,  axis=0)
    return grad.tolist()