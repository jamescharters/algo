import sys
import dill
import numpy as np
import argparse
from lda_cgs import LDA

def read_train_data(filepath):
    data = np.genfromtxt(filepath, dtype=float)
    header = data[0,:-1].astype(int)
    indices = data[1:,:-1].astype(int)
    data = data[1:,-1].astype(int)
    return (header,indices,data)

p = argparse.ArgumentParser()
p.add_argument("-i", "--infile", help="input matrix x", type=argparse.FileType('r'), required=True)
p.add_argument("-o", "--outfile", help="output model file", type=argparse.FileType('w'), required=True)
p.add_argument("-k", help="k", type=int, required=True)
p.add_argument("--max_iter", help="Maximum number of iterations", type=int, default='30')
p.add_argument("-v", "--verbose", help="verbosity", action='store_true')
args = p.parse_args()

header,indices,data= read_train_data(args.infile)
nusers = header[0]
nsongs = header[1]

X = []
for i in range(nusers):
    X.append([])

for index in indices:
    u = index[0]
    s = index[1]
    X[u].append(s)

alpha = np.array([0.1]*args.k)
beta = np.array([0.1]*nsongs)

model = LDA(alpha,beta,args.max_iter,args.verbose)
model.fit(X)
dill.dump(model, args.outfile)
