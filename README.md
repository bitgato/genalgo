# genalgo

C++ implementations of SGA and NSGA-II for solving multi-objective optimization
problem. The benchmark function used by default is SCH2.

## Simple Genetic Algorithm

SGA with naive weighted-sum approach.

### Compiling

`g++ -O3 -Werror -Wall -Wextra -Wpedantic -std=c++14 -o sga sga.cpp`

## Non-Dominated Sorting Genetic Algorithm II

NSGA-II is implemented as outlined in the original research paper by K. Deb et
al.

### Compiling

`g++ -O3 -Werror -Wall -Wextra -Wpedantic -std=c++14 -o nsga nsga.cpp`

## Functions

The problems on the which the algorithms run can be changed by changing the
contents of the `SGAIndividual::calc_objs()` and
`NSGAIndividual::calc_objs()` functions.
