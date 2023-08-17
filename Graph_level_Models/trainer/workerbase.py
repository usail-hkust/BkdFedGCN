import logging
import torch
import time
from abc import ABCMeta, abstractmethod

from Graph_level_Models.helpers.metrics import accuracy_TU as accuracy
import os
logger = logging.getLogger('client.workerbase')

'''
This is the worker for sharing the local weights.
'''
class WorkerBase(metaclass=ABCMeta):
    def __init__(self, model, loss_func, train_iter, attack_iter, test_iter, config, optimizer, device):
        self.model = model
        self.loss_func = loss_func

        self.train_iter = train_iter
        self.test_iter = test_iter
        self.attack_iter = attack_iter
        self.config = config
        self.optimizer = optimizer

        # Accuracy record
        self.acc_record = [0]

        self.device = device
        self._level_length = None
        self._weights_len = 0
        self._weights = None

    def get_weights(self):
        """ getting weights """
        return self._weights

    def set_weights(self, weights):
        """ setting weights """
        self._weights = weights

    def upgrade(self):
        """ Use the processed weights to update the model """

        idx = 0
        for param in self.model.parameters():

            tmp = self._weights[self._level_length[idx]:self._level_length[idx + 1]]
            weights_re = torch.tensor(tmp, device=self.device)
            weights_re = weights_re.view(param.data.size())

            param.data = weights_re
            idx += 1

    @abstractmethod
    def update(self):
        pass

    ## GNN model training:

    def gnn_train(self,global_model,args): # This function is for local train one epoch using local dataset on client
        """ General local training methods """
        self.model.train()
        self.acc_record = [0]
        train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        #self.optimizer.zero_grad()
        for batch_graphs, batch_labels in self.train_iter:

            batch_graphs = batch_graphs.to(self.device)
            batch_x = batch_graphs.ndata['feat'].to(self.device)  # num x feat
            batch_e = batch_graphs.edata['feat'].to(self.device)

            batch_labels = batch_labels.to(torch.long)
            batch_labels = batch_labels.to(self.device)
            self.optimizer.zero_grad()
            batch_scores = self.model.forward(batch_graphs, batch_x, batch_e)
            l = self.model.loss(batch_scores, batch_labels)
            if args.defense == "fedprox":
                # compute proximal_term
                proximal_term = 0.0
                for w, w_t in zip(self.model.parameters(), global_model.parameters()):
                    proximal_term += (w - w_t).norm(2)

                l = l + (args.mu / 2) * proximal_term


            l.backward()
            self.optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += accuracy(batch_scores, batch_labels)
            n += batch_labels.size(0)
            batch_count += 1

        self._weights = []
        self._level_length = [0]

        for param in self.model.parameters():
            self._level_length.append(param.data.numel() + self._level_length[-1])
            self._weights += param.data.view(-1).cpu().numpy().tolist()
        self._weights_len = len(self._weights)

        # print train acc of each client
        if self.attack_iter is not None:
            test_acc, test_l,  att_acc = self.gnn_evaluate()
        else:
            test_acc, test_l = self.gnn_evaluate()
        return train_l_sum / batch_count, train_acc_sum / n, test_l, test_acc

    def gnn_evaluate(self):
        acc_sum, acc_att, n, test_l_sum = 0.0, 0.0, 0, 0.0
        batch_count = 0
        self.model.eval()
        with torch.no_grad():
            for batch_graphs, batch_labels in self.test_iter:

                batch_graphs = batch_graphs.to(self.device)

                batch_x = batch_graphs.ndata['feat'].to(self.device)
                batch_e = batch_graphs.edata['feat'].to(self.device)

                batch_labels = batch_labels.to(torch.long)
                batch_labels = batch_labels.to(self.device)
                batch_scores = self.model.forward(batch_graphs, batch_x, batch_e)
                l = self.loss_func(batch_scores, batch_labels)
                acc_sum += accuracy(batch_scores, batch_labels)
                test_l_sum += l.detach().item()
                n += batch_labels.size(0)
                batch_count += 1
                if self.attack_iter is not None:
                    n_att = 0
                    self.model.eval()
                    for batch_graphs, batch_labels in self.attack_iter:
                        batch_graphs = batch_graphs.to(self.device)

                        batch_x = batch_graphs.ndata['feat'].to(self.device)
                        batch_e = batch_graphs.edata['feat'].to(self.device)
                        batch_labels = batch_labels.to(self.device)
                        batch_scores = self.model.forward(batch_graphs, batch_x, batch_e)
                        acc_att += accuracy(batch_scores, batch_labels)
                        #self.model.train()
                        n_att += batch_labels.size(0)
                    return acc_sum / n, test_l_sum / batch_count, acc_att / n_att

        return acc_sum / n, test_l_sum / batch_count
 
