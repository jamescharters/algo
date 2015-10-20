import random
import numpy as np
from scipy.sparse import lil_matrix

class LDA:
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def fit(self, X,k,t=1000):
        z = self.initialize_z(X, k)
        theta = self.initialize_theta(X, k)
        phi = self.initialize_phi(X, k)
        i = 0
        while True:
            z_new = self.update_z(X, theta, phi)
            theta_new = self.update_theta(z)
            phi_new = self.update_phi(X, z)
            z = z_new
            theta = theta_new
            phi = phi_new
            i += 1
            if t <= i: break
        return (z,theta,phi)

    def initialize_z(self, X, k):
        z = []
        for d in range(len(X)):
            z.append(np.random.randint(low=0,high=k,size=len(X[d])))
        return z

    def initialize_theta(self, X, k):
        theta = np.random.dirichlet(alpha=self.alpha, size=len(X))
        return theta

    def initialize_phi(self, X, k):
        phi = np.random.dirichlet(alpha=self.beta, size=k)
        return phi

    def update_z(self, X, theta, phi):
        z = []
        for i in range(len(X)): # d
            z.append([])
            for j in range(len(X[i])): # i
                prob = []
                for k in range(phi.shape[0]):
                    prob.append(phi[k,X[i][j]]*theta[i,k])
                prob = np.array(prob)
                prob = prob/prob.sum()
                a = np.random.multinomial(n=1, pvals=prob, size=1)
                z[i].append(np.argmax(np.random.multinomial(n=1, pvals=prob, size=1)))
        return z

    def update_theta(self, z):
        n = []
        for i in range(len(z)): # d
            n.append([])
            for k in range(self.alpha.shape[0]):
                n[i].append(np.sum(np.array(z[i])==k))
        theta = []
        for d in range(len(z)):
            theta.append(np.random.dirichlet(alpha=n[d]+self.alpha, size=1))
        theta = np.vstack(theta)
        return theta

    def update_phi(self, X, z):
        n = []
        stacked_X = np.hstack(X)
        stacked_z = np.hstack(z)
        for k in range(self.alpha.shape[0]):
            n.append([])
            for v in range(self.beta.shape[0]):
                n[k].append(np.sum((stacked_X==v)&(stacked_z==k)))
        n = np.array(n)
        phi = []
        for k in range(self.alpha.shape[0]):
            phi.append(np.random.dirichlet(alpha=n[k]+self.beta, size=1))
        phi = np.vstack(phi)
        return phi

if __name__ == '__main__':
    X = [[0,0,1,1,2,2],[0,1,2],[3,4],[0,1,1,1,2],[3,4,4],[3,3,4,4],[0,0,4,4]]
    n_topics = 2
    n_vocab = 5
    alpha = np.array([0.1]*n_topics)
    beta= np.array([0.1]*n_vocab)
    lda = LDA(alpha=alpha, beta=beta)
    z, theta, phi = lda.fit(X,len(alpha), 1000)
    print X
    print z
    print theta
    print phi
