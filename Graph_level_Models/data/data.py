"""
    File to load dataset based on user control from main file
"""
import os
#os.chdir('../') # go to root folder of the pro
import sys
sys.path.append('/home/nfs/federated_learning_jx/federated_learning/GNN_common/data')

from TUs import TUsDataset


def LoadData(DATASET_NAME):
    """
        This function is called in the main.py file 
        returns:
        ; dataset object
    """    

    # handling for the TU Datasets
    """DATASET_NAME, such as ``ENZYMES``, ``DD``, ``COLLAB``, ``MUTAG``, can be the
        datasets name on `<https://chrsmrrs.github.io/datasets/docs/datasets/>`_.
    """
    return TUsDataset(DATASET_NAME)
