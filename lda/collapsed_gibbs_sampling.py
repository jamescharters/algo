import random
import numpy as np
from scipy.sparse import lil_matrix

class LDA:
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def fit(self,X,k,t=1000):
        z = self.initialize_z(X, k)
        i = 0
        while True:
            ndk = self.update_ndk(z)
            nkv = self.update_nkv(X, z)
            z_new = self.update_z(X, z, ndk, nkv)
            z = z_new
            i += 1
            if t <= i: break
        return z

    def initialize_z(self, X, k):
        z = []
        for d in range(len(X)):
            z.append(np.random.randint(low=0,high=k,size=len(X[d])))
        return z

    def update_ndk(self, z):
        ndk = []
        for d in range(len(z)):
            ndk.append([])
            for k in range(self.alpha.shape[0]):
                ndk[d].append(np.sum(np.array(z[d])==k))
        return ndk

    def update_nkv(self, X, z):
        nkv = []
        stacked_X = np.hstack(X)
        stacked_z = np.hstack(z)
        for k in range(self.alpha.shape[0]):
            nkv.append([])
            for v in range(self.beta.shape[0]):
                nkv[k].append(np.sum((stacked_X==v)&(stacked_z==k)))
        return nkv

    def update_z(self, X, old_z, ndk, nkv):
        nd = []
        for d in range(len(X)):
            nd.append(.0)
            for k in range(self.alpha.shape[0]):
                nd[d] += ndk[d][k] + self.alpha[k]
        nk = []
        for k in range(self.alpha.shape[0]):
            nk.append(.0)
            for v in range(self.beta.shape[0]):
                nk[k] += nkv[k][v] + self.beta[v]

        z = []
        for d in range(len(X)): # d
            z.append([])
            for i in range(len(X[d])): # i
                v = X[d][i]
                prob = []
                for k in range(self.alpha.shape[0]):
                    if old_z[d][i] == k:
                        prob.append(((nkv[k][v]-1+self.beta[v])/(nk[k]-1))*((ndk[d][k]-1+self.alpha[k])/(nd[d]-1)))
                    else:
                        prob.append(((nkv[k][v]+self.beta[v])/(nk[k]-1))*((ndk[d][k]+self.alpha[k])/(nd[d]-1)))
                prob = np.array(prob)
                prob = prob/prob.sum()
                z[d].append(np.argmax(np.random.multinomial(n=1, pvals=prob, size=1)))
        return z

if __name__ == '__main__':
    X = [[0,0,1,1,2,2],[0,1,2],[3,4],[0,1,1,1,2],[3,4,4],[3,3,4,4],[0,0,4,4]]
    n_topics = 2
    n_vocab = 5
    alpha = np.array([0.1]*n_topics)
    beta= np.array([0.1]*n_vocab)
    lda = LDA(alpha=alpha, beta=beta)
    z = lda.fit(X,len(alpha), 1000)
    print X
    print z
