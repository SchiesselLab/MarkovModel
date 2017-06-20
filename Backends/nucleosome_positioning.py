# Author: Marco Tompitak, marcotompitak.github.io

# This package provides the backend that performs the
# calculations for the Markov-chain model of
# Tompitak et al. BMC Bioinformatics 18 (2017) 157.
#
# The functionality of this backend depends on input
# values for the probabilities of the relevant oligo-
# nucleotides. Two such sets of probabilities must be
# loaded in, one being the probabilities of oligo-
# nucleotides of length n (the 'long' probabilities),
# and one for those of length n-1 ('short'), n being 
# the order of the model. See the article referenced 
# above for more information.

import numpy as np
import sys
import math

class NPBackend:
  """
  Main class providing the calculation backend. Needs to be initialized
  with keywords
  
  order      -- model order (i.e. 3 for trinucleotide, 2 for dinucleotide, etc.)
  seq_length -- number of base pairs in the system
  Pl         -- probability tensor for the 'long' oligonucleotides
  Ps         -- if order > 1, probability tensor for the 'short' oligonucleotides
  """
  bound_bp_right = [7,18,30,39,50,60,70,81,91,101,112,122,132,144]
  bound_bp_left = [2,14,24,34,45,55,65,76,86,96,107,116,128,139]
  
  def __init__(self, order, seq_length, Pl, Ps=None):
    self.order = order
    self.seq_length = seq_length
    self.Pl = Pl
    self.Ps = Ps
    if ( order < 1 ):
      print("ERROR: order cannot be smaller than 1!", file=sys.stderr)
      exit(1)
    if ( order > 1 and Ps is None ):
      print("ERROR: order > 1 but no Ps specified!", file=sys.stderr)
      exit(1)
  
  @staticmethod
  def StrToSeq(string):
    """
    Converts a string of A, T, C and G characters into a sequence object.
    """
    if isinstance(string, np.ndarray):
      return string
    arr = np.empty(len(string)).astype(int)
    for i in range(len(string)):
      if ( string[i] == 'A' or string[i] == 'a' ):
        arr[i] = 0
      elif ( string[i] == 'T' or string[i] == 't' ):
        arr[i] = 1
      elif ( string[i] == 'C' or string[i] == 'c' ):
        arr[i] = 2
      elif ( string[i] == 'G' or string[i] == 'g' ):
        arr[i] = 3
      else:
        arr[i] = 0 # For now, undefined nucleotides replaced with A. Beware bias.
    return arr
  
  @staticmethod
  def SeqToStr(seq):
    """
    Converts a sequence object into a string of A, T, C and G characters.
    """
    arr = []
    for i in range(len(seq)):
        if seq[i] == 0:
            arr.append('A')
        elif seq[i] == 1:
            arr.append('T')
        elif seq[i] == 2:
            arr.append('C')
        elif seq[i] == 3:
            arr.append('G')
        else:
            arr.append('X')
    return "".join(arr)
  
  @staticmethod
  def OTup(sequence, pos, order):
    """
    Generates the lookup tuple to extract the right probability from the
    probability tensors, based on the oligonucleotide sequence.
    
    Arguments:
    sequence -- larger sequence that contains the oligonucleotide of interest
    pos      -- position of the oligonucleotide of interest within the sequence
    order    -- size of the oligonucleotide of interest
    """
    otup = (pos,)
    for i in range(order):
      otup += (int(sequence[pos+i]),)
    return otup
  
  @staticmethod
  def OTupOffset(sequence, pos, order, offset):
    """
    Generates the lookup tuple in the presence of an offset to the
    positions along the sequence. Useful when the input sequence is
    automatically padded with additional nucleotides, e.g. in the
    case of partially unwrapped nucleosomes.
    
    Arguments:
    sequence -- larger sequence that contains the oligonucleotide of interest
    pos      -- position of the oligonucleotide of interest within the sequence
    order    -- size of the oligonucleotide of interest
    offset   -- offset by which to translate pos
    """
    otup = (pos+offset,)
    for i in range(order):
      otup += (sequence[pos+i],)
    return otup
  
  def Prob(self, sequence):
    """
    Calculate the probability of a full-length sequence.
    
    Arguments:
    sequence -- sequence to which to apply the calculation.
    """
    
    # This special case is handled by a separate function
    if ( self.order == 1):
      return self.ProbMono(sequence)
    
    # We strictly enforce the sequence length because it must match
    # the size of the probability tensors
    if ( len(sequence) != self.seq_length ):
      print("ERROR: sequence not the right length!", file=sys.stderr)
      exit(1)
    
    # Calculate the first factor in the multiplication
    otups = self.OTup(sequence, 0, self.order-1)
    if (self.Ps[otups] == 0.0):
      prob = 0.25
    else:
      prob = self.Ps[otups]
    
    # Loop through the rest of the multiplication
    for n in range(self.order-1, self.seq_length):
      otupl = self.OTup(sequence, n+1-self.order, self.order)
      otups = self.OTup(sequence, n+1-self.order, self.order-1)
      
      # Depending on how the probability tensors were generated,
      # some oligonucleotides may be too rare to have an estimated
      # probability that is non-zero. In this case we take this
      # as a failure to measure the correct probability, and we
      # assume a uniform distribution.
      #   This is probably not optimal, and could potentially be
      # improved.
      if (self.Pl[otupl] == 0.0 or self.Ps[otups] == 0.0):
        #print("WARNING: Zero probability encountered!", file=sys.stderr)
        pfac = 0.25
      
      # If all is well, multiply our probability with the next factor.
      else:
        pfac = self.Pl[otupl]/self.Ps[otups]
      prob *= pfac
    return prob
  
  def Prob_Unwrapped(self, sequence, first_bound_site, last_bound_site):
    if ( self.order == 1):
      return self.ProbMono_Unwrapped(sequence, first_bound_site, last_bound_site)
      
    if (first_bound_site == 0):
      bp_start = 0
    else:
      bp_start = NPBackend.bound_bp_left[first_bound_site-1]
    if (last_bound_site == 14):
      bp_end = 146
    else:
      bp_end = NPBackend.bound_bp_right[last_bound_site-1]
    #print(bp_start, bp_end)
    otups = self.OTup(sequence, bp_start, self.order-1)
    #print(otups)
    if (self.Ps[otups] == 0.0):
      prob = 0.25
    else:
      prob = self.Ps[otups]
      
    for n in range(bp_start+self.order-1, bp_end+1):
      otupl = self.OTup(sequence, n+1-self.order, self.order)
      otups = self.OTup(sequence, n+1-self.order, self.order-1)
      if (self.Pl[otupl] == 0.0 or self.Ps[otups] == 0.0):
        #print("WARNING: Zero probability encountered!", file=sys.stderr)
        pfac = 0.25
      else:
        pfac = self.Pl[otupl]/self.Ps[otups]
      prob *= pfac
    return prob
  
  def ProbMono(self, sequence):
    """
    Calculate the probability of a full-length sequence, given
    a model of order 1. This function is completely analogous to
    the Prob function, but at order 1 there are no 'long' and
    'short' oligonucleotides involved, so the calculation is
    slightly different.
    
    Arguments:
    sequence -- sequence to which to apply the calculation.
    """
    
    # Check the length of the sequence.
    if ( len(sequence) != 147 ):
      print("ERROR: sequence not 147 bp long!", file=sys.stderr)
      exit(1)
    
    # Calculate the first factor/
    otupl = self.OTup(sequence, 0, self.order)
    if (self.Pl[otupl] == 0.0):
      prob = 0.25
    else:
      prob = self.Pl[otupl]
      
    # Loop through the rest.
    for n in range(self.order-1, 147):
      otupl = self.OTup(sequence, n+1-self.order, self.order)
      if (self.Pl[otupl] == 0.0):
        pfac = 0.25
      else:
        pfac = self.Pl[otupl]
      prob *= pfac
    return prob
  
  def ProbMono_Unwrapped(sequence, first_bound_site, last_bound_site):
      
    bp_start = NPBackend.bound_bp_left[first_bound_site-1]
    bp_end = NPBackend.bound_bp_right[last_bound_site-1]
    
    otupl = self.OTup(sequence, bp_start, self.order)
    if (self.Pl[otupl] == 0.0):
      prob = 0.25
    else:
      prob = self.Pl[otupl]
      
    for n in range(bp_start+self.order-1, bp_end+1):
      otupl = self.OTup(sequence, n+1-self.order, self.order)
      if (self.Pl[otupl] == 0.0):
        pfac = 0.25
      else:
        pfac = self.Pl[otupl]
      prob *= pfac
    return prob
  
  def ProbSmoothed(self, sequence):
    """
    NOT FOR PRODUCTION
    
    Calculate the probability of a full-length sequence, but
    instead of assuming a uniform distribution if a probability
    tensor element is zero, all probabilities are smoothed, as
    in Segal et al. (2006). See Tompitak et al. BMC Bioinformatics
    18 (2017) 157 for more information.
    
    We showed that this is less optimal than using the method
    employed in the Prob function, so do not use this function
    for further applications.
    """
    if ( len(sequence) != self.seq_length ):
      print("ERROR: sequence not the right seq_length!", file=sys.stderr)
      exit(1)
    otups1 = self.OTupOffset(sequence, 0, self.order-1, 0)
    otups2 = self.OTupOffset(sequence, 0, self.order-1, 1)
    prob = 0.5*(self.Ps[otups1]+self.Ps[otups2])
    if ( prob == 0.0 ):
      prob = 0.0625
    
    otupl1 = self.OTupOffset(sequence, 0, self.order, 0)
    otupl2 = self.OTupOffset(sequence, 0, self.order, 1)
    pfac = 0.5*(self.Pl[otupl1]/self.Ps[otups1]+self.Pl[otupl2]/self.Ps[otups2])
    if ( pfac == 0.0 ):
      pfac = 0.25
    prob *= pfac
    
    for n in range(self.order, 146):
      otupl0 = self.OTupOffset(sequence, n+1-self.order, self.order, -1)
      otupl1 = self.OTupOffset(sequence, n+1-self.order, self.order, 0)
      otupl2 = self.OTupOffset(sequence, n+1-self.order, self.order, 1)
      otups0 = self.OTupOffset(sequence, n+1-self.order, self.order-1, -1)
      otups1 = self.OTupOffset(sequence, n+1-self.order, self.order-1, 0)
      otups2 = self.OTupOffset(sequence, n+1-self.order, self.order-1, 1)
      pfac = (self.Pl[otupl0]/self.Ps[otups0] + self.Pl[otupl1]/self.Ps[otups1] + self.Pl[otupl2]/self.Ps[otups2])/3.0
      if ( pfac == 0.0 ):
        pfac = 0.25
      prob *= pfac
    
    pfac = 0.5*(self.Pl[otupl1]/self.Ps[otups1]+self.Pl[otupl2]/self.Ps[otups2])
    if ( pfac == 0.0 ):
      pfac = 0.25
    prob *= pfac
    
    return prob
  
  def ProbMonoSmoothed(self, sequence):
    """
    NOT FOR PRODUCTION
    
    Analogous to ProbMono, but with the smoothing of ProbSmoothed.
    """
    if ( len(sequence) != 147 ):
      print("ERROR: sequence not 147 bp long!", file=sys.stderr)
      exit(1)
    
    otupl1 = self.OTupOffset(sequence, 0, self.order, 0)
    otupl2 = self.OTupOffset(sequence, 0, self.order, 1)
    prob = 0.5*(self.Pl[otupl1]+self.Pl[otupl2])
    if ( prob == 0.0 ):
      prob = 0.25
    
    for n in range(self.order, 146):
      otupl0 = self.OTupOffset(sequence, n+1-self.order, self.order, -1)
      otupl1 = self.OTupOffset(sequence, n+1-self.order, self.order, 0)
      otupl2 = self.OTupOffset(sequence, n+1-self.order, self.order, 1)
      pfac = (self.Pl[otupl0] + self.Pl[otupl1] + self.Pl[otupl2])/3.0
      if ( pfac == 0.0 ):
        pfac = 0.25
      prob *= pfac
    
    pfac = 0.5*(self.Pl[otupl1]+self.Pl[otupl2])
    if ( pfac == 0.0 ):
      pfac = 0.25
    prob *= pfac
    
    return prob

  def ProbLandscape(self, longseq):
    """
    Calculate the probability landscape along a sequence with at least
    as many nucleotides as the system size. The function will calculate
    a probability at each possible position for the system to sit along
    the longer sequence.
    
    Arguments:
    longseq -- the long sequence along which to calculate the landscape
    
    Returns:
    prob    -- an array containing the probability values
    """
    if ( len(longseq) < self.seq_length ):
      print("ERROR: sequence too short!", file=sys.stderr)
      exit(1)
    N = len(longseq) - self.seq_length + 1
    prob = np.zeros((N,))
    for i in range(N):
      if ( self.order > 1 ):
        prob[i] = self.Prob(self.StrToSeq(longseq[i:i+self.seq_length]))
      else:
        prob[i] = self.ProbMono(self.StrToSeq(longseq[i:i+self.seq_length]))
    return prob
  
  def PadSequence(self, seq, first_bound_site, last_bound_site):
    """
    Pad a sequence with undefined dinucleotides. Useful for partially
    unwrapped nucleosomes, or tetramers.
    """
    bp_start = NPBackend.bound_bp_left[first_bound_site-1]
    bp_end = NPBackend.bound_bp_right[last_bound_site-1]
    
    padding_left = np.empty((bp_start,))
    padding_right = np.empty((146-bp_end,))
    padded_seq = np.concatenate((padding_left, seq, padding_right))
    return padded_seq

  def ProbLandscape_Unwrapped(self, longseq, first_bound_site, last_bound_site):
    """
    As ProbLandscape, for a partially unwrapped nucleosome. 
    """
    
    # The sequence of interest needs to be padded, because our system
    # size is still 147, but we want to be able to move the partially 
    # unwrapped nucleosome closer to the edge of the sequence, as
    # allowed by the fact that not the entire system need be wrapped
    # with DNA.
    bp_start = NPBackend.bound_bp_left[first_bound_site-1]
    bp_end = NPBackend.bound_bp_right[last_bound_site-1]
    padded_seq = self.PadSequence(self.StrToSeq(longseq), first_bound_site, last_bound_site)
    
    # The rest of the calculation continues as normal.
    N = len(padded_seq) - 146
    prob = np.zeros((N,))
    for i in range(N):
      if ( self.order > 1 ):
        prob[i] = self.Prob_Unwrapped(self.StrToSeq(padded_seq[i:i+147]), first_bound_site, last_bound_site)
      else:
        prob[i] = self.ProbMono_Unwrapped(self.StrToSeq(padded_seq[i:i+147]))
    return prob
    
  
  def ProbSmoothedLandscape(self, longseq):
    """
    NOT FOR PRODUCTION
    
    As ProbLandscape, using the Smoothed functions.
    """
    if ( len(longseq) < self.seq_length ):
      print("ERROR: sequence too short!", file=sys.stderr)
      exit(1)
    N = len(longseq) - self.seq_length + 1
    prob = np.zeros((N,))
    for i in range(N):
      if ( self.order > 1 ):
        prob[i] = self.ProbSmoothed(self.StrToSeq(longseq[i:i+self.seq_length]))
      else:
        prob[i] = self.ProbMonoSmoothed(self.StrToSeq(longseq[i:i+self.seq_length]))
    return prob
  
  def EnergyLandscape(self, longseq, beta):
    """
    Returns the free energy landscape along a sequence from the
    probability landscape, at a given temperature T, encoded by
    the inverse temperature beta = 1/(k*T), where k is the Boltzmann
    constant.
    """
    if ( len(longseq) < self.seq_length ):
      print("ERROR: sequence too short!", file=sys.stderr)
      exit(1)
    
    # Calculate the probability landscape
    prob = self.ProbLandscape(longseq)
    
    # Convert to free energy landscape
    E = -np.log(prob)/beta
    return E

  def EnergyLandscape_Unwrapped(self, longseq, beta, first_bound_site, last_bound_site):
    """
    As EnergyLandscape, but for partially unwrapped nucleosomes.
    """
    prob = self.ProbLandscape_Unwrapped(longseq, first_bound_site, last_bound_site)
    E = -np.log(prob)/beta
    return E
