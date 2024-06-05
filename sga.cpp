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

const std::string sga_file = "sga_populations";
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

// parameters for SGA (provided by user)
parameters params;

// weights for each objective function (in order)
std::vector<double> WTS;

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
	WTS = std::vector<double>(params.NOBJ);
	std::cout << "Number of constraints: ";
	std::cin >> params.NCON;
}

// ---------------------------------------------------------------------------80

class
SGAIndividual
{
public:
	// the values of variables
	std::vector<double> vars;
	// the values of objective functions
	std::vector<double> objs;
	// the values of constraint penalties
	std::vector<double> cons;
	// fitness of the individual (relative to the population)
	double fitness;
	// sum of absolute values of constraint violations
	// note that constraints are to be converted to the form:
	// g(x1...xn) >= 0 for the program to work correctly
	double cons_violation = 0.0;

	SGAIndividual();
	SGAIndividual(const std::vector<double>& vars);

	void calc_objs();
	void flip_bit(int v, int i);
	bool better_than(const SGAIndividual& j) const;
	void display(std::ostream& os) const;

	bool operator == (const SGAIndividual& j) const;
};

SGAIndividual::SGAIndividual() {
	fitness = 0.0;
	vars = std::vector<double>(params.NVAR);
	objs = std::vector<double>(params.NOBJ);
	cons = std::vector<double>(params.NCON);
}

SGAIndividual::SGAIndividual(const std::vector<double>& vars) {
	fitness = 0.0;
	this->vars = vars;
	calc_objs();
}

void
SGAIndividual::calc_objs() {
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
SGAIndividual::flip_bit(int v, int i) {
	// flips bit at i'th position
	double xl = params.LIMS[v].XL;
	double step = params.STEPS[v];
	int xb = get_xb(vars[v], xl, step);
	xb = xb ^ (1UL << i);
	vars[v] = get_x(xb, xl, step);
}

bool
SGAIndividual::better_than(const SGAIndividual& j) const {
	// returns true if this individual is better than given individual j
	// from two infeasible solutions, the one with lower
	// constraint violation is preferred
	if(cons_violation > 0 && j.cons_violation > 0) {
		return (cons_violation < j.cons_violation);
	}
	// a feasible solution is preferred to an infeasible solution
	if(cons_violation == 0 && j.cons_violation > 0) {
		return true;
	}
	if(cons_violation > 0 && j.cons_violation == 0) {
		return false;
	}
	return fitness > j.fitness;
}

void
SGAIndividual::display(std::ostream& os) const {
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
SGAIndividual::operator == (const SGAIndividual& j) const {
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
SGAPopulation
{
public:
	std::vector<SGAIndividual> popn;
	int size;

	SGAPopulation();

	void initialize();
	void add(const SGAIndividual& individual);
	void sort();
	void resize();
	void copy_from(const SGAPopulation& p);
	int select_parent_index(std::vector<double>& wts);
	void crossover(int p1_ind, int p2_ind, SGAPopulation& child_popn);
	void mutate(int ind);
	void calc_fitnesses();
	void calc_wts(std::vector<double>& wts);
	void gen_next_popn();
	void display(std::ostream& os) const;
};

SGAPopulation::SGAPopulation() {
	size = 0;
}

void
SGAPopulation::initialize() {
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
		add(SGAIndividual(vars));
	}
}

void
SGAPopulation::add(const SGAIndividual& individual) {
	popn.push_back(individual);
	size++;
}

void
SGAPopulation::sort() {
	// sorts the population such that fitter individuals are at the top
	struct sort_comp {
		bool operator() (const SGAIndividual& i, const SGAIndividual& j) {
			return i.better_than(j);
		}
	};
	std::sort(popn.begin(), popn.end(), sort_comp());
}

void
SGAPopulation::resize() {
	// resize the population to NP individuals
	// takes top NP individuals, so sort before resizing
	popn.resize(params.NP);
	size = params.NP;
}

void
SGAPopulation::copy_from(const SGAPopulation& p) {
	popn = p.popn;
	size = p.size;
}

int
SGAPopulation::select_parent_index(std::vector<double>& wts) {
	// selects a parent's index based on given weights
	std::discrete_distribution<> d(wts.begin(), wts.end());
	int p_ind = d(rand_engine);
	// sets the selected individual's weight to 0
	// so that it won't get selected as second parent in the same round
	wts[p_ind] = 0;
	return p_ind;
}

void
SGAPopulation::calc_fitnesses() {
	// calculates scaled fitness values for all individuals in this population
	for(auto& i : popn) {
		i.fitness = 0.0;
	}
	// scaling objectives between 0 to 1
	for(int o=0; o<params.NOBJ; o++) {
		// the scaled values for oth objective function
		std::vector<double> scaled_objs(size);
		// minimum value of oth objective in this population
		double min_obj = popn[0].objs[o];
		// maximum value of oth objective in this population
		double max_obj = min_obj;
		scaled_objs[0] = min_obj;
		for(int i=1; i<size; i++) {
			scaled_objs[i] = popn[i].objs[o];
			if(scaled_objs[i] < min_obj) min_obj = scaled_objs[i];
			if(scaled_objs[i] > max_obj) max_obj = scaled_objs[i];
		}
		// note that ith member of scaled_objs is the oth objective value
		// fot ith individual in this population
		for(int i=0; i<size; i++) {
			// scale the objectives from 0 to 1
			scaled_objs[i] = (scaled_objs[i] - min_obj) / (max_obj - min_obj);
			// this makes the individuals with lower objective values have
			// higher scaled objective values
			scaled_objs[i] = 1.0 - scaled_objs[i];
			// update fitness for individuals by adding weight of o'th
			// objective multiplied by objective value of ith individual
			popn[i].fitness += (WTS[o] * scaled_objs[i]);
		}
	}
}

void
SGAPopulation::calc_wts(std::vector<double>& wts) {
	// calculate weights for selection of each individual in the population
	// a fitter individual is more likely to get selected
	double total_fitness = 0.0;
	for(auto& i : popn) {
		total_fitness += i.fitness;
	}
	for(int i=0; i<size; i++) {
		wts[i] = popn[i].fitness / total_fitness;
	}
}

void
SGAPopulation::crossover(int p1_ind, int p2_ind, SGAPopulation& child_popn) {
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
	child_popn.add(SGAIndividual(c1vars));
	child_popn.add(SGAIndividual(c2vars));
}

void
SGAPopulation::mutate(int ind) {
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
SGAPopulation::gen_next_popn() {
	// create a child population and add all parents to it
	SGAPopulation child_popn(*this);

	std::vector<double> wts(size, 0);
	calc_wts(wts);
	// NP/2 crossovers for NP parents
	for(int i=0; i<size/2; i++) {
		auto tmp_wts = wts;

		// when tmp_wts is passed for selecting P1, its weight is set to zero
		// so when tmp_wts is used for selecting P2, P1 will not be selected
		int p1 = select_parent_index(tmp_wts);
		int p2 = select_parent_index(tmp_wts);

		// check if crossover will happen or not
		bool cross = gen_prob(params.PC);
		// crossover won't happen, so selecting is useless as well
		if(!cross) continue;
		crossover(p1, p2, child_popn);

		// mutate the newly added children
		child_popn.mutate(child_popn.size - 1);
		child_popn.mutate(child_popn.size - 2);
	}

	child_popn.calc_fitnesses();
	// sort the child population from best to worst
	child_popn.sort();
	// select best Np individuals from child population
	child_popn.resize();
	// replace current population with child population
	this->copy_from(child_popn);
}

void
SGAPopulation::display(std::ostream& os) const {
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

// -----------------------------------------------------------------------------

int
main() {
	std::ofstream of;
	std::cout << std::endl;
	get_params();
	// take input for weights
	for(int i=0; i<params.NOBJ; i++) {
		std::cout << "Weight for objective " << (i + 1) << ": ";
		std::cin >> WTS[i];
	}
	int num_sols;
	// take input for number of solutions needed
	std::cout << "Number of solutions to find for feasible population: ";
	std::cin >> num_sols;

	SGAPopulation feasible_popn;

	of.open(sga_file);
	for(int i=1; i<=num_sols; i++) {
		SGAPopulation p;
		p.initialize();
		p.calc_fitnesses();

		of << "Solution " << i << " : Generation 1" << std::endl;

		// write the population to a file
		p.display(of);

		for(int j=2; j<=params.MAX_GEN; j++) {
			of << "Solution " << i << " : Generation " << j << std::endl;

			p.gen_next_popn();
			// write the population to a file
			p.display(of);
		}

		of << "Solution "<< i << " :";
		for(int v=0; v<params.NVAR; v++) {
			of
			<< " var" + std::to_string(v+1)
			<< " = "
			<< std::fixed << std::setprecision(COLP)
			<< p.popn[0].vars[v];
		}
		of << std::endl << std::endl;

		// add the best solution to the feasible population
		feasible_popn.add(p.popn[0]);
	}
	of.close();

	std::cout << std::endl;
	std::cout << std::left << "Feasible Population" << std::endl;
	feasible_popn.display(std::cout);

	if(params.TOPLOT) {
		// write the feasible population to a file for plotting
		of.open(feasible_file);
		feasible_popn.display(of);
		of.close();
	}

	return 0;
}

