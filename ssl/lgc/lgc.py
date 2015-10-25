import numpy as np
from scipy.sparse import diags

class LGC:
    def __init__(self,alpha):
        self.alpha = alpha

    def fit(self,Y,W,t=300):
        """ iter """
        F = np.zeros(Y.shape)
        D2 = np.sqrt(diags((1.0/(W.sum(1))).T.tolist()[0],offsets=0))
        S = D2.dot(W).dot(D2)

        i = 0
        while True:
            F = (1-self.alpha)*Y + self.alpha*S.dot(F)
            if i > t: break
            i += 1

        self.F = F

    def predict(self,node_list):
        return np.argmax(self.F[node_list],axis=1)

    def predict_proba(self,node_list):
        return (self.F[node_list].T / np.sum(self.F[node_list], axis=1)).T

if __name__ == '__main__':
    from scipy.sparse import lil_matrix
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

    Y = lil_matrix((4,2))
    Y[0,1]=1.
    Y[2,0]=0.9
    Y[2,1]=0.1

    clf = LGC(alpha=0.99)
    clf.fit(Y,G)
    print G.todense()
    print clf.predict(np.array([0,1,2,3]))
    print clf.predict_proba(np.array([0,1,2,3]))
