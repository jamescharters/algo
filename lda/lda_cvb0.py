import random
import numpy as np
from scipy.sparse import lil_matrix
from scipy.special import digamma

class LDA:
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def fit(self, X,k,t=1000):
        qz = self.initialize_qz(X)
        Enkv = self.initialize_Enkv(qz,X)
        Endk = self.initialize_Endk(qz,X)
        i = 0
        while True:
            new_qz = self.update_qz(qz, X, Enkv, Endk)
            Enkv = self.update_Enkv(new_qz, qz, X, Enkv)
            Endk = self.update_Endk(new_qz, qz, X, Endk)
            qz = new_qz
            i += 1
            if t <= i: break
        return (qz,Enkv,Endk)

    def initialize_qz(self,X):
        qz = {}
        for d in range(len(X)):
            qz[d] = {}
            for i in range(len(X[d])):
                qz[d][i] = {}
                sum_qzdi = 0
                for k in range(self.alpha.shape[0]):
                    qz[d][i][k] = random.random()
                    sum_qzdi += qz[d][i][k]
                for k in range(self.alpha.shape[0]):
                    qz[d][i][k] /= sum_qzdi
        return qz

    def initialize_Enkv(self,qz,X):
        Enkv = {}
        for k in range(self.alpha.shape[0]):
            Enkv[k] = {}
            for v in range(self.beta.shape[0]):
                Enkv[k][v] = 0
        for d in range(len(X)):
            for i in range(len(X[d])):
                v = X[d][i]
                for k in range(self.alpha.shape[0]):
                    Enkv[k][v] += qz[d][i][k]
        return Enkv

    def initialize_Endk(self,qz,X):
        Endk = {}
        for d in range(len(X)):
            Endk[d] = {}
            for k in range(self.alpha.shape[0]):
                Endk[d][k] = 0
                for i in range(len(X[d])):
                    Endk[d][k] += qz[d][i][k]
        return Endk


    def update_qz(self, old_qz, X, Enkv, Endk):
        Enk = {}
        for k in range(self.alpha.shape[0]):
            Enk[k] = 0
            for v in range(self.beta.shape[0]):
                Enk[k] += Enkv[k][v]
        sum_beta = self.beta.sum()
        qz = {}
        for d in range(len(X)):
            qz[d] = {}
            for i in range(len(X[d])):
                qz[d][i] = {}
                sum_qzdi = 0
                v = X[d][i]
                for k in range(self.alpha.shape[0]):
                    qz[d][i][k] = (Enkv[k][v]-old_qz[d][i][k] + self.beta[v]) * (Endk[d][k]-old_qz[d][i][k] + self.alpha[k]) / (Enk[k]-old_qz[d][i][k] + sum_beta)
                    sum_qzdi += qz[d][i][k]
                for k in range(self.alpha.shape[0]):
                    qz[d][i][k] /= sum_qzdi
        return qz

    def update_Enkv(self, new_qz, qz, X, Enkv):
        for d in range(len(X)):
            for i in range(len(X[d])):
                v = X[d][i]
                for k in range(self.alpha.shape[0]):
                    Enkv[k][v] = Enkv[k][v] - qz[d][i][k] + new_qz[d][i][k]
        return Enkv

    def update_Endk(self, new_qz, qz, X, Endk):
        for d in range(len(X)):
            for i in range(len(X[d])):
                for k in range(self.alpha.shape[0]):
                    Endk[d][k] = Endk[d][k] - qz[d][i][k] + new_qz[d][i][k]
        return Endk

if __name__ == '__main__':
    X = [[0,0,1,1,2,2],[0,1,2],[3,4],[0,1,1,1,2],[3,4,4],[3,3,4,4],[0,0,4,4]]
    n_topics = 2
    n_vocab = 5
    alpha = np.array([0.1]*n_topics)
    beta= np.array([0.1]*n_vocab)
    lda = LDA(alpha=alpha, beta=beta)
    qz, Enkv, Endk= lda.fit(X,len(alpha), 1000)
    z = []
    for d in range(len(qz)):
        z.append([])
        for i in range(len(qz[d])):
            z[d].append(sorted(qz[d][i].items(),key=lambda x:x[1],reverse=True)[0][0])
    print X
    print z
    for d in range(len(qz)):
        for k in range(len(alpha)):
            print Endk[d][k],
        print
    for k in range(len(alpha)):
        for v in range(len(beta)):
            print Enkv[k][v],
        print
