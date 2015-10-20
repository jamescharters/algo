import random
import numpy as np
from scipy.sparse import lil_matrix
from scipy.special import digamma

class LDA:
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def fit(self, X,k,t=1000):
        Enkv = self.initialize_Enkv()
        Endk = self.initialize_Endk(X)
        i = 0
        while True:
            qz = self.update_qz(X, Enkv, Endk)
            Enkv = self.update_Enkv(qz, X)
            Endk = self.update_Endk(qz, X)
            i += 1
            if t <= i: break
        return (qz,Enkv,Endk)

    def initialize_Enkv(self):
        Enkv = {}
        for k in range(self.alpha.shape[0]):
            Enkv[k] = {}
            for v in range(self.beta.shape[0]):
                Enkv[k][v] = random.randint(0,10) # tekitou
        return Enkv

    def initialize_Endk(self,X):
        Endk = {}
        for d in range(len(X)):
            Endk[d] = {}
            for k in range(self.alpha.shape[0]):
                Endk[d][k] = random.randint(0,10) # tekitou
        return Endk


    def update_qz(self, X, Enkv, Endk):
        qz = {}
        End = {}
        for d in range(len(X)):
            End[d] = sum(Endk[d].values())
        Enk = {}
        for k in range(self.alpha.shape[0]):
            Enk[k] = sum(Enkv[k].values())
        for d in range(len(X)):
            qz[d] = {}
            for i in range(len(X[d])): # i
                qz[d][i] = {}
                v = X[d][i]
                for k in range(self.alpha.shape[0]):
                    qz[d][i][k] = (np.exp(digamma(Endk[d][k] + self.alpha[k])) / np.exp(digamma(End[d]+sum(self.alpha)))) * (np.exp(digamma(Enkv[k][v] + self.beta[v])) / np.exp(digamma(Enk[k]+sum(self.beta))))
                qzdi_sum = sum(qz[d][i].values())
                for k in range(self.alpha.shape[0]):
                    qz[d][i][k] /= qzdi_sum
        return qz

    def update_Enkv(self, qz, X):
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

    def update_Endk(self, qz, X):
        Endk = {}
        for d in range(len(X)):
            Endk[d] = {}
            for k in range(self.alpha.shape[0]):
                Endk[d][k] = 0
        for d in range(len(X)):
            for i in range(len(X[d])):
                for k in range(self.alpha.shape[0]):
                    Endk[d][k] += qz[d][i][k]
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
