# Autonomous Graph Mining Algorithm Search with Best Speed/Accuracy Trade-off

AUTOGM is an automated system for graph mining algorithm development. 

We first define a unified framework UNIFIEDGM that integrates various message-passing based graph algorithms, ranging from conventional
algorithms like PageRank to graph neural networks.
UNIFIEDGM defines a search space in which five parameters are required to determine a graph algorithm. 
Under this search space, AUTOGM explicitly optimizes for the optimal parameter set of UNIFIEDGM using Bayesian Optimization. 
AUTOGM defines a novel budget-aware objective function for the optimization to incorporate a practical issue — finding the best speed-accuracy trade-off under a computation budget — into the
graph algorithm generation problem.

Details can be found in the original paper (https://minjiyoon.xyz/Paper/AutoGM.pdf)
