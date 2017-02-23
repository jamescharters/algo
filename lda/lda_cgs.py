import random
import numpy as np
from scipy.sparse import lil_matrix

class LDA:
    def __init__(self, alpha, beta, max_iter, verbose=0):
        self.alpha = alpha
        self.beta = beta
        self.asum = self.alpha.sum()
        self.bsum = self.beta.sum()
        self.max_iter = max_iter
        self.verbose=verbose

    def fit(self,X):
        self.N = len(X) # number of documents
        self.K = self.alpha.shape[0] # number of topics
        self.V = self.beta.shape[0] # number of vocabulary

        self.z = self._init_z(X)
        self.ndk, self.nkv = self._init_params(X)

        remained_iter = self.max_iter
        while True:
            if self.verbose: print remained_iter
            nk = self.nkv.sum(axis=1) # k-dimensional vector O(KV)
            for d in range(self.N):
                for i in range(len(X[d])):
                    k = self.z[d][i]
                    v = X[d][i]

                    self.ndk[d][k] -= 1
                    self.nkv[k][v] -= 1
                    nk[k] -= 1

                    self.z[d][i] = self._sample_z(X,d,v,nk)

                    self.ndk[d][self.z[d][i]] += 1
                    self.nkv[self.z[d][i]][v] += 1
                    nk[self.z[d][i]] += 1
            remained_iter -= 1
            if remained_iter <= 0: break
        return self

    def _init_z(self,X):
        z = []
        for d in range(len(X)):
            z.append(np.random.randint(low=0,high=self.K,size=len(X[d])))
        return z

    def _init_params(self,X):
        ndk = np.zeros((self.N,self.K)) + self.alpha
        nkv = np.zeros((self.K,self.V)) + self.beta
        for d in range(self.N):
            for i in range(len(X[d])):
                k = self.z[d][i]
                v = X[d][i]
                ndk[d,k]+=1
                nkv[k,v]+=1
        return ndk,nkv

    def _sample_z(self,X,d,v,nk):
        ndk = self.ndk[d,:] # k-dimensional vector
        nkv = self.nkv[:,v] # k-dimensional vector

        prob = (ndk+self.alpha) *  ((nkv+self.beta[v])/(nk+self.bsum))
        prob = prob/prob.sum()
        z = np.random.multinomial(n=1, pvals=prob).argmax()
        return z

if __name__ == '__main__':
    X = [[0,0,1,1,2,2],[0,1,2],[3,4],[0,1,1,1,2],[3,4,4],[3,3,4,4],[0,0,4,4]]
    n_topics = 2
    n_vocab = 5
    alpha = np.array([0.1]*n_topics)
    beta= np.array([0.1]*n_vocab)
    lda = LDA(alpha=alpha, beta=beta, max_iter=100, verbose=1)
    lda.fit(X)
    print X
    print lda.z
    print lda.ndk
    print lda.nkv

