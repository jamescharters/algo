import numpy as np
from scipy.sparse import lil_matrix
from scipy.misc import logsumexp
class MRF:
    def __init__(self,phi,psi,G):
        """ Computed in log space """
        self.phi = self.safe_log(phi) # node potentials
        self.psi = self.safe_log(psi) # edge potentials
        self.G = G
        self.L = psi.shape[0]  # number of different labels

    def safe_log(self,x,xclip=0.00000000001):
        return np.log(x.clip(xclip))

    def error(self,m1,m2):
        err = sum([abs(m1[k]-m2[k]) for k in range(self.L)]).sum()
        n = sum([m1[k].nnz for k in range(self.L)])
        return err / n

    def fit(self, th=0.001):
        m = {}
        for k in range(self.L):
            m[k] = self.G.copy()
        while True:
            new_m = {}
            m_prod = np.zeros((self.G.shape[0],self.L))
            for k in range(self.L):
                m_prod[:,k] = m[k].sum(axis=0)
            sum_mk = []
            for k in range(self.L):
                sum_mk.append(logsumexp(self.phi+self.psi[:,k]+m_prod,axis=1))
                new_m[k] = self.G.multiply(lil_matrix(sum_mk[-1])).T
            sum_m = self.G.multiply(lil_matrix(logsumexp(sum_mk, axis=0))).T
            for k in range(self.L):
                new_m[k] -= sum_m # normalization
            if self.error(new_m,m) < th:
                break
            else:
                m = new_m
        self.m = new_m

    def predict(self,nid):
        b = self.phi[nid] + np.array([self.m[k][:,nid].sum() for k in range(self.L)])
        p = b - logsumexp(b)
        max_label = np.argmax(p)
        return max_label,np.exp(p[max_label])

if __name__ == '__main__':
  G = lil_matrix((4,4))
  G[0,1]=1
  G[0,2]=1
  G[1,0]=1
  G[1,2]=1
  G[2,0]=1
  G[2,1]=1
  G[2,3]=1
  G[3,2]=1
  G.tocsr()

  phi = np.zeros((4,2))
  phi[0,1]=1.
  phi[1,0]=0.5
  phi[1,1]=0.5
  phi[2,0]=0.9
  phi[2,1]=0.1
  phi[3,0]=0.5
  phi[3,1]=0.5

  psi = np.zeros((2,2))
  psi[0,0] = 0.51
  psi[0,1] = 0.49
  psi[1,0] = 0.49
  psi[1,1] = 0.51

  clf = MRF(phi,psi,G)
  clf.fit()
  print phi
  print G.todense()
  print clf.predict(0)
  print clf.predict(1)
  print clf.predict(2)
  print clf.predict(3)
