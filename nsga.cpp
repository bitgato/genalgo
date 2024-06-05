#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

// random seed from clock for the random engine
unsigned seed = std::chrono::steady_clock::now().time_since_epoch().count();
// engine for producing random numbers using the Mersenne twister algorithm
std::mt19937 rand_engine(seed);

const std::string nsga_file = "nsga_populations";
const std::string feasible_file = "feasible_population";

struct
limits {
	// lower limit of search interval
	double XL;
	// upper limit of search interval
	double XU;
};

struct
parameters {
	// number of variables
	int NVAR;
	// number of bits in encoding of a variable
	std::vector<int> NS;
	// search interval for a variable
	std::vector<limits> LIMS;
	// step size in the search interval (calculated from N)
	std::vector<double> STEPS;
	// number of members in a population
	int NP;
	// number of generations to stop after
	int MAX_GEN;
	// probability of crossover
	double PC;
	// probability of mutation
	double PM;
	// number of objectives
	int NOBJ;
	// number of constraints
	int NCON;
	// whether to write the feasible population to a file or not
	// the feasible population will be written to STDOUT regardless
	bool TOPLOT;
};

// display column width and precision parameters
int COLW = 16, COLP = 8;

// parameters for NSGA (provided by user)
parameters params;

// ---------------------------------------------------------------------------80

bool
gen_prob(double p) {
	// generates true with given probability p
	std::bernoulli_distribution d(p);
	return d(rand_engine);
}

int
get_xb(double x, double xl, double step) {
	// returns the xb value for a given value of x (in integer form)
	double xb_d = (x - xl) / step;
	int xb = static_cast<int>(round(xb_d));
	return xb;
}

double
get_x(int xb, double xl, double step) {
	// returns the x value for a given value of xb (in double form)
	double x = xl + (xb * step);
	return x;
}

void
get_params() {
	params.TOPLOT = true;
	std::cout
	<< "Do you want the feasible population in a file for plotting? (Y/n): ";
	char ans = 0;
	std::cin >> ans;
	if(ans == 'n' || ans == 'N') {
		params.TOPLOT = false;
	}

	std::cout << "Number of variables: ";
	std::cin >> params.NVAR;
	for(int i=0; i<params.NVAR; i++) {
		std::cout << "Number of bits in encoding of var" << (i+1) << ": ";
		int n;
		std::cin >> n;
		params.NS.push_back(n);
		limits lim;
		std::cout << "Lower limit for var" << (i+1) << ": ";
		std::cin >> lim.XL;
		std::cout << "Upper limit for var" << (i+1) << ": ";
		std::cin >> lim.XU;
		params.LIMS.push_back(lim);
		// step size in the interval if the binary encoding is n-bits
		double step = (lim.XU - lim.XL) / ((1UL << n) - 1);
		params.STEPS.push_back(step);
	}
	std::cout << "Number of members in a population: ";
	std::cin >> params.NP;
	std::cout << "Number of generations to stop after: ";
	std::cin >> params.MAX_GEN;
	std::cout << "Probability of crossover: ";
	std::cin >> params.PC;
	std::cout << "Probability of mutation: ";
	std::cin >> params.PM;
	std::cout << "Number of objectives: ";
	std::cin >> params.NOBJ;
	std::cout << "Number of constraints: ";
	std::cin >> params.NCON;
}

// ---------------------------------------------------------------------------80

class
NSGAIndividual
{
public:
	// the values of variables
	std::vector<double> vars;
	// the values of objective functions
	std::vector<double> objs;
	// the values of constraint penalties
	std::vector<double> cons;
	// rank of the individual
	int rank = 0;
	// crowding distance of the individual
	double crowding_dist = 0.0;
	// sum of absolute values of constraint violations
	// note that constraints are to be converted to the form:
	// g(x1...xn) >= 0 for the program to work correctly
	double cons_violation = 0.0;

	NSGAIndividual();
	NSGAIndividual(const std::vector<double>& vars);

	void calc_objs();
	void flip_bit(int v, int i);
	bool dominates(const NSGAIndividual& j) const;
	bool wins_against(const NSGAIndividual& j) const;
	void display(std::ostream& os) const;

	bool operator == (const NSGAIndividual& j) const;
};

NSGAIndividual::NSGAIndividual() {
	vars = std::vector<double>(params.NVAR);
	objs = std::vector<double>(params.NOBJ);
	cons = std::vector<double>(params.NCON);
}

NSGAIndividual::NSGAIndividual(const std::vector<double>& vars) {
	this->vars = vars;
	calc_objs();
}

void
NSGAIndividual::calc_objs() {
	objs = std::vector<double>(params.NOBJ);

	double x = vars[0];
	if(x <= 1) {
		objs[0] = -x;
	}
	else if(x <= 3) {
		objs[0] = x - 2;
	}
	else if(x <= 4) {
		objs[0] = 4 - x;
	}
	else {
		objs[0] = x - 4;
	}
	objs[1] = (x - 5) * (x - 5);

	for(auto& c : cons) {
		if(c < 0) cons_violation += std::abs(c);
	}
}

void
NSGAIndividual::flip_bit(int v, int i) {
	// flips bit at i'th position
	double xl = params.LIMS[v].XL;
	double step = params.STEPS[v];
	int xb = get_xb(vars[v], xl, step);
	xb = xb ^ (1UL << i);
	vars[v] = get_x(xb, xl, step);
}

bool
NSGAIndividual::dominates(const NSGAIndividual& j) const {
	// returns true if this individual dominates given individual j
	// from two infeasible solutions, the one with lower
	// constraint violation is preferred
	if(cons_violation > 0 && j.cons_violation > 0) {
		return (cons_violation > j.cons_violation);
	}
	// a feasible solution is preferred to an infeasible solution
	if(cons_violation == 0 && j.cons_violation > 0) {
		return true;
	}
	if(cons_violation > 0 && j.cons_violation == 0) {
		return false;
	}
	// from two feasible solutions, the one with better
	// objective functions is preferred
	bool worse = false, better = false;
	for(int o=0; o<params.NOBJ; o++) {
		// i is better than j in atleast one objective
		if(objs[o] < j.objs[o]) {
			better = true;
		}
		// i is worse than j in atleast one objective
		else if(objs[o] > j.objs[o]) {
			worse = true;
		}
	}
	// this individual dominates given individual j if this individual is
	// not worse than j in any of the objectives, and is better than j in
	// atleast one objective
	return (!worse && better);
}

bool
NSGAIndividual::wins_against(const NSGAIndividual& j) const {
	// returns true if this individual wins a tournament against given
	// individual j

	// crowding distance is considered only when ranks are same
	if(rank == j.rank) {
		// if the ranks are same and this individual has a higher crowding
		// distance, then it wins a tournament against the given individual j
		return (crowding_dist > j.crowding_dist);
	}
	// if this individual has a higher rank, it wins a tournament
	return (rank < j.rank);
}

void
NSGAIndividual::display(std::ostream& os) const {
	// prints the individual's variables and objective function values
	// to the output stream provided (can be a file or stdout)
	std::ios init(nullptr);
	init.copyfmt(os);

	for(auto& v : vars) {
		os << std::fixed
		<< std::setprecision(COLP)
		<< std::setw(COLW)
		<< std::right << v;
	}
	for(auto& f : objs) {
		os << std::fixed
		<< std::setprecision(COLP)
		<< std::setw(COLW)
		<< std::right << f;
	}
	os << std::endl;

	os.copyfmt(init);
}

bool
NSGAIndividual::operator == (const NSGAIndividual& j) const {
	// this operator is overloaded so that the std::find method works
	// two individuals are considered equal
	// if all their variables are equal
	for(int v=0; v<params.NVAR; v++) {
		if(vars[v] != j.vars[v]) return false;
	}
	return true;
}

// ---------------------------------------------------------------------------80

class
NSGAPopulation
{
public:
	std::vector<NSGAIndividual> popn;
	int size;

	NSGAPopulation();

	void initialize();
	void add(const NSGAIndividual& individual);
	void add(const NSGAPopulation& individuals);
	void sort();
	void resize();
	void copy_from(const NSGAPopulation& p);
	int select_parent_index(std::vector<int>& tmp_wts, std::vector<int>& wts);
	void crossover(int p1_ind, int p2_ind, NSGAPopulation& child_popn);
	void mutate(int ind);
	void calc_ranks_crowding();
	void gen_child_popn(NSGAPopulation& child_popn);
	void gen_next_popn();
	void display(std::ostream& os) const;
};

NSGAPopulation::NSGAPopulation() {
	size = 0;
}

void
NSGAPopulation::initialize() {
	// initialize NP random individuals
	while(size < params.NP) {
		std::vector<double> vars(params.NVAR);
		for(int v=0; v<params.NVAR; v++) {
			std::uniform_int_distribution<> d(0, 1UL << params.NS[v]);
			// choose random value of xb
			int xb = d(rand_engine);
			double xl = params.LIMS[v].XL;
			double step = params.STEPS[v];
			vars[v] = get_x(xb, xl, step);
		}
		add(NSGAIndividual(vars));
	}
}

void
NSGAPopulation::add(const NSGAIndividual& individual) {
	popn.push_back(individual);
	size++;
}

void
NSGAPopulation::add(const NSGAPopulation& individuals) {
	// adds individuals from given population
	popn.insert(
		popn.end(),
		std::make_move_iterator(individuals.popn.begin()),
		std::make_move_iterator(individuals.popn.end())
	);
	size = popn.size();
}

void
NSGAPopulation::sort() {
	// sorts the population using the crowding tournament operator, placing
	// better individuals first.
	struct sort_comp {
		bool operator() (const NSGAIndividual& i, const NSGAIndividual& j) {
			return i.wins_against(j);
		}
	};
	std::sort(popn.begin(), popn.end(), sort_comp());
}

void
NSGAPopulation::resize() {
	// resize the population to NP individuals
	// takes top NP individuals, so sort before resizing
	popn.resize(params.NP);
	size = params.NP;
}

void
NSGAPopulation::copy_from(const NSGAPopulation& p) {
	popn = p.popn;
	size = p.size;
}

int
NSGAPopulation::select_parent_index(std::vector<int>& tmp_wts,
					std::vector<int>& wts) {

	// Selects a prent index based on weights provided.
	// Epoch: All parents are randomly selected for exactly two tournaments.
	// Round: Four parents are randomly selected and the two winners of the
	//        tournaments are used for crossover.
	std::discrete_distribution<> d(tmp_wts.begin(), tmp_wts.end());
	int p_ind = d(rand_engine);

	// reduce the temporary weight to 0
	// so that this parent won't get selected in the same round
	tmp_wts[p_ind] = 0;

	// reduce the weight in further iterations
	// if this weight goes to zero, it will not get
	// selected for further tournaments in this epoch
	wts[p_ind]--;

	return p_ind;
}

void
NSGAPopulation::crossover(int p1_ind, int p2_ind, NSGAPopulation& child_popn) {
	std::vector<double> c1vars(params.NVAR), c2vars(params.NVAR);
	// crossover all variables
	for(int v=0; v<params.NVAR; v++) {
		int N = params.NS[v];
		// generate random crossover point in interval (1, N-2)
		// the positions before 1st bit and after last bit
		// are not in the selection pool
		std::uniform_int_distribution<> d(1, N - 1);
		int xpt = d(rand_engine);

		double xl = params.LIMS[v].XL;
		double step = params.STEPS[v];

		int p1_xb = get_xb(popn[p1_ind].vars[v], xl, step);
		int p2_xb = get_xb(popn[p2_ind].vars[v], xl, step);
		// mask for first part
		int fm = ((1UL << N) - 1) << (N - xpt);
		// mask for second part
		int sm = ((1UL << N) - 1) >> xpt;
		// first and second parts of parent 1
		int p1_f = (p1_xb & fm), p1_s = (p1_xb & sm);
		// first and second parts of parent 1
		int p2_f = (p2_xb & fm), p2_s = (p2_xb & sm);

		// compbine first part of parent 1 with second part of parent 2
		int c1 = p1_f | p2_s;
		c1vars[v] = get_x(c1, xl, step);

		// compbine first part of parent 2 with second part of parent 1
		int c2 = p2_f | p1_s;
		c2vars[v] = get_x(c2, xl, step);
	}
	child_popn.add(NSGAIndividual(c1vars));
	child_popn.add(NSGAIndividual(c2vars));
}

void
NSGAPopulation::mutate(int ind) {
	// mutates the individual at given index
	// iterate through the variables
	for(int v=0; v<params.NVAR; v++) {
		// iterate through the bits of the variable
		for(int i=0; i<params.NS[v]; i++) {
			// if true is generated (with probability PM), flip the bit
			if(gen_prob(params.PM)) {
				popn[ind].flip_bit(v, i);
			}
		}
	}
}

void
NSGAPopulation::calc_ranks_crowding() {
	// calculates ranks and crowding distances for individuals

	double inf = std::numeric_limits<double>::infinity();
	// maximum and minimum objectives in the population
	std::vector<double> max_objs(params.NOBJ, -inf);
	std::vector<double> min_objs(params.NOBJ, inf);
	for(int o=0; o<params.NOBJ; o++) {
		for(auto& i : popn) {
			if(i.objs[o] > max_objs[o]) max_objs[o] = i.objs[o];
			if(i.objs[o] < min_objs[o]) min_objs[o] = i.objs[o];
		}
	}

	// whether i'th individual is deleted or not
	std::vector<bool> deleted(size, false);
	// total number of deleted individuals
	int c_deleted = 0;

	NSGAPopulation rankedpopn;
	// non-dominated fronts are made till all individuals are deleted
	for(int rank = 1; c_deleted < size; rank++) {
		// current non-dominated front
		NSGAPopulation ndf;
		std::vector<int> ndf_inds;
		for(int i=0; i < size; i++) {
			if(deleted[i]) continue;
			bool dominated = false;
			for(int j=0; j < size; j++) {
				if((j == i) || deleted[j]) continue;
				if(popn[j].dominates(popn[i])) {
					dominated = true;
					break;
				}
			}
			if(!dominated) {
				popn[i].rank = rank;
				ndf_inds.push_back(i);
			}
		}
		for(auto& i : ndf_inds) {
			ndf.add(popn[i]);
			deleted[i] = true;
			c_deleted++;
		}

		// for every objective
		for(int o=0; o<params.NOBJ; o++) {
			// sort the non-dominated front in worse order
			std::sort(ndf.popn.begin(), ndf.popn.end(),
				[o](const auto& i, const auto& j) {
				return (i.objs[o] < j.objs[o]);
			});

			// most separated individuals are given highest crowding distance
			ndf.popn[0].crowding_dist = inf;
			ndf.popn[ndf.size - 1].crowding_dist = inf;

			// calculate crowding distances for rest of the individuals
			for(int i=1; i<ndf.size - 1; i++) {
				if(std::isinf(ndf.popn[i].crowding_dist)) continue;
				// numerator
				double num = ndf.popn[i+1].objs[o] - ndf.popn[i-1].objs[o];
				// denominator
				double den = max_objs[o] - min_objs[o];
				// update crowding distance
				ndf.popn[i].crowding_dist += (num / den);
			}
		}

		// and add the individuals to ranked population
		rankedpopn.add(ndf);
	}
	// replace this population by ranked population
	copy_from(rankedpopn);
	// sort the population according to crowding tournament operator
	sort();
}

void
NSGAPopulation::gen_child_popn(NSGAPopulation& child_popn) {
	// overall weights according to which parents are selected in an epoch
	// this is maintained so that we can keep track of how many tournaments
	// an individual has participated in
	// if this weight reduces to zero (individual has participated in two
	// tournaments), it will not get selected for another tournament
	std::vector<int> wts(size, 2);
	// note that we are selecting 4 parents in a round
	// so the population size NP should be divisible by 4
	for(int num_tmnts=0; num_tmnts < size/2; num_tmnts++) {
		// copy the epoch weights as temporary weights for each round
		std::vector<int> tmp_wts = wts;
		// select four parents for two tournaments
		int ind1, ind2, ind3, ind4;
		int p1, p2;
		ind1 = select_parent_index(tmp_wts, wts);
		ind2 = select_parent_index(tmp_wts, wts);
		ind3 = select_parent_index(tmp_wts, wts);
		ind4 = select_parent_index(tmp_wts, wts);
		// use the winners for crossover
		if(popn[ind1].wins_against(popn[ind2])) p1 = ind1;
		else p1 = ind2;
		if(popn[ind3].wins_against(popn[ind4])) p2 = ind3;
		else p2 = ind4;

		// crossover
		bool cross = gen_prob(params.PC);
		if(!cross) continue;
		crossover(p1, p2, child_popn);

		// mutate the newly added children
		child_popn.mutate(child_popn.size - 1);
		child_popn.mutate(child_popn.size - 2);
	}
}

void
NSGAPopulation::gen_next_popn() {
	// generates child population Qt
	NSGAPopulation q, r;
	gen_child_popn(q);
	// combined population Rt
	r.add(*this);
	r.add(q);
	r.calc_ranks_crowding();
	r.sort();
	r.resize();
	// replace this population by Rt for next iteration (t+1)
	copy_from(r);
}

void
NSGAPopulation::display(std::ostream& os) const {
	// displays the population to the given output stream
	std::ios init(nullptr);
	init.copyfmt(os);

	os << std::setfill('#')
	<< std::setw((params.NOBJ + params.NVAR) * COLW) << ""
	<< std::endl;
	os << std::setfill(' ');

	for(int v=1; v<=params.NVAR; v++) {
		os << std::setw(COLW)
		<< std::left << "var" + std::to_string(v);
	}
	for(int i=1; i<=params.NOBJ; i++) {
		os << std::setw(COLW)
		<< std::left << "f" + std::to_string(i);
	}
	os << std::endl;

	os << std::setfill('#')
	<< std::setw((params.NOBJ + params.NVAR) * COLW) << ""
	<< std::endl;
	os << std::setfill(' ');

	for(auto& i : popn) {
		i.display(os);
	}

	os.copyfmt(init);
}

// ---------------------------------------------------------------------------80

int
main() {
	std::ofstream of;
	std::cout << std::endl << "NOTE: The number of members in a "
	<< "population should be divisible by 4" << std::endl;
	std::cout << std::endl;
	get_params();

	of.open(nsga_file);
	NSGAPopulation p, q;
	p.initialize();
	p.calc_ranks_crowding();
	of << "Generation 1" << std::endl;
	p.display(of);

	for(int i=2; i<=params.MAX_GEN; i++) {
		p.gen_next_popn();
		of << "Generation " << i << std::endl;
		p.display(of);
	}
	of.close();

	std::cout << std::endl;
	std::cout << "Feasible Population" << std::endl;
	p.display(std::cout);

	if(params.TOPLOT) {
		of.open(feasible_file);
		p.display(of);
		of.close();
	}

	return 0;
}

