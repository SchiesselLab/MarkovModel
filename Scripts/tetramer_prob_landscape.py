# Author: Marco Tompitak, marcotompitak.github.io
# Produced for the group of Prof. Helmut Schiessel
# at Leiden University, schiessellab.github.io

# This script makes use of the nucleosome_positioning
# backend to calculate the probability landscape 
# for a tetramer along a sequence. The tetramer is
# implemented as a partially unwrapped nucleosome.

import numpy as np
import sys
import math

sys.path.append('../Backends/')
from nucleosome_positioning import NPBackend

if ( not (len(sys.argv) == 5 or (len(sys.argv) == 4 and int(sys.argv[1]) == 1)) ):
        print("Usage: python3 nucleosome_prob_landscape.py <order> <seqfile> <filelong> <fileshort>")
        print("<order> is the length of the longest oligonucleotides to take into account,")
        print("i.e. 2 for dinucleotides.")
        print("<seqfile> is the path to the file that contains the sequence to analyze.")
        print("<filelong> is the path to the file that contains probability distributions")
        print("for the oligonicleotides of length <order>")
        print("<fileshort> is the path to the file that contains probability distributions")
        print("for the oligonicleotides of length <order>-1")
        exit(0)

# Load in the command line arguments
order = int(sys.argv[1])
seqfile = sys.argv[2]
filelong = sys.argv[3]

# Load in the probability tensor for the 'long' oligonucleotides
rshptuplong = (148-order,) + (4,)*order
Pl = np.genfromtxt(filelong).reshape(rshptuplong)

# Read in the sequence
with open (seqfile, "r") as myfile:
  data=myfile.read()
Seq = "".join(data.split())

if ( order > 1 ):
  # If order > 1, we also need to load in the probability
  # tensor for the 'short' oligonucleotides
  fileshrt = sys.argv[4]
  rshptupshrt = (149-order,) + (4,)*(order-1)
  Ps = np.genfromtxt(fileshrt).reshape(rshptupshrt)
  # Set up the backend
  NP = NPBackend(order, 147, Pl, Ps)
else:
  NP = NPBackend(order, 147, Pl)

# Calculate the landscape
p = NP.ProbLandscape_Unwrapped(Seq, 5, 10)

# Output
np.savetxt(sys.stdout.buffer,p)
