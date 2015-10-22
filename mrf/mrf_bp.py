class MRF:
  def __init__(self,phi,psi,G,L):
    self.phi = phi # node potentials
    self.psi = psi # edge potentials
    self.G = G
    self.GR = self.make_GR(G)
    self.L = L  # number of different labels

  def make_GR(self,G):
    GR = {}
    for i in G:
      for j in G[i]:
        if not j in GR: GR[j] = set([])
        GR[j].add(i)
    return GR

  def error(self,m1,m2):
    err = 0.
    n = 0
    for i in m1:
      for j in m1[i]:
        n += 1
        for k in range(self.L):
          err += abs(m1[i][j][k]-m2[i][j][k])
    return err / n

  def fit(self, th=0.001):
    m = {}
    for i in self.G:
      m[i] = {}
      for j in self.G[i]:
        m[i][j] = {}
        for l in range(self.L): # number of labels
          m[i][j][l] = 1.0 / self.L
    while True:
      new_m = {}
      for i in self.G:
        new_m[i] = {}
        m_prod = {}
        for s in range(self.L):
          m_prod[s] = 1
          for k in self.GR[i]:
            m_prod[s] *= m[k][i][s]
        for j in self.G[i]:
          new_m[i][j] = {}
          for l in range(self.L):
            new_m[i][j][l] = 0
            for s in range(self.L):
              new_m[i][j][l] += self.phi[i][s]*self.psi[s][l]*m_prod[s]
          sum_m = float(sum(new_m[i][j].values()))
          for l in range(self.L):
            new_m[i][j][l] /= sum_m
      if self.error(new_m,m) < th:
        break
      else:
        m = new_m
    self.m = new_m

  def predict(self,nid):
    max_b = -1
    max_label = -1
    sum_b = 0.
    for k in range(self.L):
      b = self.phi[nid][k]
      for j in self.GR[nid]:
        b *= self.m[j][nid][k]
      sum_b += b
      if b > max_b:
        max_b = b
        max_label = k
    return max_label,max_b/sum_b

if __name__ == '__main__':
  G = {1:set([2,3]),2:set([1,3]),3:set([1,2,4]),4:set([3])}
  phi = {1:{0:0.,1:1.},2:{0:0.5,1:0.5},3:{0:0.9,1:0.1},4:{0:0.5,1:0.5}}
  psi = {0:{0:0.51,1:0.49}, 1:{0:0.49,1:0.51}}
  clf = MRF(phi,psi,G,2)
  clf.fit()
  print clf.predict(1)
  print clf.predict(2)
  print clf.predict(3)
  print clf.predict(4)
