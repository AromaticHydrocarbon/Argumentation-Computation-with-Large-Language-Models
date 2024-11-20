/*!
 * Copyright (c) <2018> <Andreas Niskanen, University of Helsinki>
 * 
 * 
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * 
 * 
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * 
 * 
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#ifndef OPENWBO_SOLVER_H
#define OPENWBO_SOLVER_H

#include "MaxSATSolver.h"
#define protected public // hack
#include "MaxSAT.h"
#undef protected
#include "algorithms/Alg_MSU3.h"
#include "MaxSATFormula.h"

class OpenWBOSolver : public MaxSATSolver {

private:

public:
	openwbo::MaxSAT * mxsolver;
	openwbo::MaxSATFormula * formula;
	openwbo::MaxSATFormula * formula_stored;
	double initial_time;
	OpenWBOSolver();
	~OpenWBOSolver() { delete formula; delete mxsolver; }
	void build_solver(int hard_weight);
	void add_hard_clause(std::vector<int> & clause);
	void add_soft_clause(int weight, std::vector<int> & clause);
	void solve();

};

#endif