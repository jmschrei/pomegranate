### Hidden Markov Models

[Hidden Markov models](http://en.wikipedia.org/wiki/Hidden_Markov_model) are a form of structured learning, in which a sequence of observations are labelled according to the hidden state they belong. HMMs can be thought of as non-greedy FSMs, in that the assignment of tags is done in a globally optimal way as opposed to being simply the best at the next step. HMMs have been used extensively in speech recognition and bioinformatics, where speech is a sequence of phonemes and DNA is a sequence of nucleotides. 

An IPython notebook tutorial with visualizations can be [found here](https://github.com/jmschrei/pomegranate/blob/master/tutorials/Tutorial_3_Hidden_Markov_Models.ipynb)

A full tutorial on sequence alignment in bioinformatics can be found [here](http://nbviewer.ipython.org/github/jmschrei/yahmm/blob/master/examples/Global%20Sequence%20Alignment.ipynb) The gist is that you have a graphical structure as follows:

![alt text](http://www.cs.tau.ac.il/~rshamir/algmb/00/scribe00/html/lec06/img106.gif "Three Character Profile HMM")

This defines a 'profile HMM' of length 3, in which you model the a profile and align new sequences to it. A perfectly matching sequence will align its three characters to the three match states. However, there can be mismatches where the observed sequence does not match the profile perfectly, which is why each emission is a distribution over all four nucleotides with small probabilities of seeing other nucleotides. There can also be insertions into the sequence, modelled by the middle track, and deleted in comparison to the profile modelled by the top track of silent states. This type of model can solve an interesting question in sequence analysis of when is it more likely that a nucleotide mutated over time (simply a mismatch in the model), versus was explicitly removed and had a new nucleotide added in (delete + insertion). We can see which is more likely using the Viterbi algorithm. 

Lets make our profile we model 'ACT'. 

```python
from pomegranate import *
model = HiddenMarkovModel( "Global Sequence Aligner" )

# Define the distribution for insertions
i_d = DiscreteDistribution( { 'A': 0.25, 'C': 0.25, 'G': 0.25, 'T': 0.25 } )

# Create the insert states, each with a uniform insertion distribution
i0 = State( i_d, name="I0" )
i1 = State( i_d, name="I1" )
i2 = State( i_d, name="I2" )
i3 = State( i_d, name="I3" )

# Create the match states with small chances of mismatches
m1 = State( DiscreteDistribution({ "A": 0.95, 'C': 0.01, 'G': 0.01, 'T': 0.02 }) , name="M1" )
m2 = State( DiscreteDistribution({ "A": 0.003, 'C': 0.99, 'G': 0.003, 'T': 0.004 }) , name="M2" )
m3 = State( DiscreteDistribution({ "A": 0.01, 'C': 0.01, 'G': 0.01, 'T': 0.97 }) , name="M3" )

# Create the silent delete states
d1 = State( None, name="D1" )
d2 = State( None, name="D2" )
d3 = State( None, name="D3" )

# Add all the states to the model
model.add_states( [i0, i1, i2, i3, m1, m2, m3, d1, d2, d3 ] )

# Create transitions from match states
model.add_transition( model.start, m1, 0.9 )
model.add_transition( model.start, i0, 0.1 )
model.add_transition( m1, m2, 0.9 )
model.add_transition( m1, i1, 0.05 )
model.add_transition( m1, d2, 0.05 )
model.add_transition( m2, m3, 0.9 )
model.add_transition( m2, i2, 0.05 )
model.add_transition( m2, d3, 0.05 )
model.add_transition( m3, model.end, 0.9 )
model.add_transition( m3, i3, 0.1 )

# Create transitions from insert states
model.add_transition( i0, i0, 0.70 )
model.add_transition( i0, d1, 0.15 )
model.add_transition( i0, m1, 0.15 )

model.add_transition( i1, i1, 0.70 )
model.add_transition( i1, d2, 0.15 )
model.add_transition( i1, m2, 0.15 )

model.add_transition( i2, i2, 0.70 )
model.add_transition( i2, d3, 0.15 )
model.add_transition( i2, m3, 0.15 )

model.add_transition( i3, i3, 0.85 )
model.add_transition( i3, model.end, 0.15 )

# Create transitions from delete states
model.add_transition( d1, d2, 0.15 )
model.add_transition( d1, i1, 0.15 )
model.add_transition( d1, m2, 0.70 ) 

model.add_transition( d2, d3, 0.15 )
model.add_transition( d2, i2, 0.15 )
model.add_transition( d2, m3, 0.70 )

model.add_transition( d3, i3, 0.30 )
model.add_transition( d3, model.end, 0.70 )

# Call bake to finalize the structure of the model.
model.bake()

for sequence in map( list, ('ACT', 'GGC', 'GAT', 'ACC') ):
    logp, path = model.viterbi( sequence )
    print "Sequence: '{}'  -- Log Probability: {} -- Path: {}".format(
        ''.join( sequence ), logp, " ".join( state.name for idx, state in path[1:-1] ) )
```

This should produce the following:

```
Sequence: 'ACT'  -- Log Probability: -0.513244900357 -- Path: M1 M2 M3
Sequence: 'GGC'  -- Log Probability: -11.0481012413 -- Path: I0 I0 D1 M2 D3
Sequence: 'GAT'  -- Log Probability: -9.12551967402 -- Path: I0 M1 D2 M3
Sequence: 'ACC'  -- Log Probability: -5.08795587886 -- Path: M1 M2 M3
```

This seems to work well. A perfect match goes through the three match states, and off matches go through other sequences of hidden states. We see in the case of ACC, the model thinks it's more likely that the T at the end of the profile was mutated to a C than a deletion and an insertion, but is not so lenient in the case of 'GGC'. This is I made G's very unlikely in our prior match distributions.

The HMM can then be used as a backend to do the alignment by defining a function which takes in two sequences, and uses the HMM in the global name space to do the alignment (works for this example, but using global name spaces in general is a bad idea). Using the HMM as a backend allows us to shield users from needing to know how HMMs work at all but only that through the power of math their sequences have been aligned.

```python
def pairwise_alignment( x, y ):
    """
    This function will take in two sequences,  and insert dashes appropriately to make them appear aligned. This consists only of adding a dash to the model sequence for every insert in the path appropriately, and a dash in the observed sequence for every delete in the path appropriately.
    """
    
    logp, path = model.viterbi( sequence )
    for i, (index, state) in enumerate( path[1:-1] ):
        name = state.name
        
        if name.startswith( 'D' ):
            y = y[:i] + '-' + y[i:]
        elif name.startswith( 'I' ):
            x = x[:i] + '-' + x[i:]

    return x, y

for sequence in map( list, ('A', 'GA', 'AC', 'AT', 'ATCC' ) ):
    x, y = pairwise_alignment( 'ACT', ''.join(sequence) )
    print "{}\n{}".format( x, y )
    print
```

yields

```
Sequence: A
ACT
A--

Sequence: GA
-ACT
GA--

Sequence: AC
ACT
AC-

Sequence: AT
ACT
A-T

Sequence: ATCC
ACT--
A-TCC
```

Everything is exactly the same as in YAHMM, except the `Model` class is now `HiddenMarkovModel`. If you have code using YAHMM you would like to port over to pomegranate, the only difference is the start of your file should have

```python
from pomegranate import *
from pomegranate import HiddenMarkovModel as Model
```

instead of

```python
from yahmm import *
```
API Reference
=============

```eval_rst
.. automodule:: pomegranate.hmm
    :members:
```