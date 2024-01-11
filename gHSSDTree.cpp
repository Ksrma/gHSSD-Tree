// greedy HSSD by reusing space partition tree: each time remove the least hypervolume contributor
// it is efficient only when k <= n-sqrt(n); complexity is O({n-k+n^(1/2)}n^{(d-1)/2}logn)

// TODO: each time remove the least contributing subset

/*-------------------------------------------------------------------

                  Copyright (c) 2023
            Jingda Deng <jddeng@xjtu.edu.cn>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

---------------------------------------------------------------------

This C++ program executes the decremental greedy hypervoluem subset
selection for dimensionality >= 3. Please refer to the following 
paper for a description of this gHSSD-Tree algorithm: 

Jingda Deng, Jianyong Sun, Qingfu Zhang, and Hui Li, "Efficient Greedy 
Hypervolume Subset Selection Using Space Partition Tree".

Compilation: g++ -O3 gHSSDTree.cpp
Usage: gHSSDTree <number of points> <dimension> <number of points to be reserved>
       <input file> <reference point file> <outputfile(optional)>
       
       Input file should contain n lines (n is the number of points).
       Each line contains d numbers separated by blanks (d is the number of dimensionality).
       
Notice: (1) Codes for timing in the main function only work in Linux. 
        Our codes can work well in Windows platform (e.g., Visual 
        Studio) after removing or changing them.
        (2) These codes work for MINIMIZATION problem, but it can be
        easily revised for MAXIMIZATION problem by changing points 
        according to the reference point when reading data file.

Special thanks to Nicola Beume for providing source codes of 
the HOY algorithm. I have learned a lot from them. 

---------------------------------------------------------------------*/

#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <vector>
#include <cmath>
#include <sys/time.h>
#include <sys/resource.h>

using namespace std;

struct BFTreeNode_largeK
{
	// data for building tree
	// LinkedList will be free after building tree
	// when k is large, dominatedCount is useful to cut off tree
	bool isQuasiPile;
	int split, usefulDims;
	int *dominatedCount;
	vector<int*> LinkedList;
	int *LinkedListSize;
	vector<int> dims;
	vector<int> quasiPiles;

	// first box and second box among all boxes that fully cover this node (they may fully cover father nodes)
	int partialCoverNumber, firstCoverBox, secondCoverBox;

	// data to maintain first and second covering boxes belonging to this node
	// that is, boxes that fully cover father node will only be recorded in father node instead of this node
	// (otherwise, the worst-case time/space complexity would be unknown)

	// data for computing contributions (valid for leaf nodes)
	bool isFirstCoverBoxInserted;
	int fullyCount, indexOfMyFirstCoverBox, indexOfMySecondCoverBox;
	int insertCount, backupInsertCount;
	double xLd, firstCoverContribution, uselessDimContribution;
	// A0 and A1 are the first and second column of Matrix A in the paper (since lambda=1 in this program)
	// if the i-th element of A0/A1 is equal to popsize, it means there is no i-pile inserted
	// otherwise, it is equal to the index of i-pile, noting that this index is
	// regarding to this node (cumulated by j in computeLeafNode() function), not the index of regarding to the whole point set.
	// backupA0 and backupA1 are the first and second column of Matrix A before the first covering box is inserted.
	int *partialCoverIndex, *fullyCoverBoxes;
	int *A0, *A1, *backupA0, *backupA1;
	double *lowerbound, *upperbound, *delta;
	double *nodeContributions;
	BFTreeNode_largeK *leftChild, *rightChild;
};

struct BFTreeNode_smallK
{
	// data for building tree
	// LinkedList will be free after building tree
	bool isQuasiPile;
	int split, usefulDims;
	vector<int*> LinkedList;
	int *LinkedListSize;
	vector<int> dims;
	vector<int> quasiPiles;

	// first box and second box among all boxes that fully cover this node (they may fully cover father nodes)
	int partialCoverNumber, firstCoverBox, secondCoverBox;

	// data to maintain first and second covering boxes belonging to this node
	// that is, boxes that fully cover father node will only be recorded in father node instead of this node
	// (otherwise, the worst-case time/space complexity would be unknown)

	// data for computing contributions (valid for leaf nodes)
	bool isFirstCoverBoxInserted;
	int fullyCount, indexOfMyFirstCoverBox, indexOfMySecondCoverBox;
	int insertCount, backupInsertCount;
	double xLd, firstCoverContribution, uselessDimContribution;
	// A0 and A1 are the first and second column of Matrix A in the paper (since lambda=1 in this program)
	// if the i-th element of A0/A1 is equal to popsize, it means there is no i-pile inserted
	// otherwise, it is equal to the index of i-pile, noting that this index is
	// regarding to this node (cumulated by j in computeLeafNode() function), not the index of regarding to the whole point set.
	// backupA0 and backupA1 are the first and second column of Matrix A before the first covering box is inserted.
	int *partialCoverIndex, *fullyCoverBoxes;
	int *A0, *A1, *backupA0, *backupA1;
	double *lowerbound, *upperbound, *delta;
	double *nodeContributions;
	BFTreeNode_smallK *leftChild, *rightChild;
};

static int dimension;
static int dimensionMinusOne;
static int alter;
static int popsize;
static int dSqrtDataNumber;
static int removedPoint;
static int reservedNumber;
static int *piles;
static int *boundaries;
static int *noBoundaries;
static int *dimensionPileCount;
static double *contributions;
static double *minusOneContributions;
static vector<bool> deleted;
static vector<double*> population;

inline bool Yildiz_cmp(double* a, double* b) {
	return (a[dimensionMinusOne] > b[dimensionMinusOne]);
}

template <typename Container>
struct compare_indirect_index_ascend
{
	const Container& container;
	const int dim;
	compare_indirect_index_ascend( const Container& container, const int dim ): container( container ), dim( dim ) { }
	bool operator () ( size_t lindex, size_t rindex ) const
	{
		return container[ lindex ][ dim ] < container[ rindex ][ dim ];
	}
};
template <typename Container>
struct compare_indirect_index_ascend2
{
	const Container& container;
	compare_indirect_index_ascend2( const Container& container ): container( container ) { }
	bool operator () ( size_t lindex, size_t rindex ) const
	{
		return container[ lindex ] < container[ rindex ];
	}
};
template <typename Container>
struct compare_indirect_index_descend
{
	const Container& container;
	const int dim;
	compare_indirect_index_descend( const Container& container, const int dim ): container( container ), dim( dim ) { }
	bool operator () ( size_t lindex, size_t rindex ) const
	{
		return container[ lindex ][ dim ] > container[ rindex ][ dim ];
	}
};

inline void Index_Ascend_Sort(vector<double*> x, int* beg, int n, int dim)
{
	std::sort(beg, beg+n, compare_indirect_index_ascend <decltype(x)> ( x, dim ) );
}
inline void Index_Ascend_Sort(vector<int> x, int* beg, int n)
{
	std::sort(beg, beg+n, compare_indirect_index_ascend2 <decltype(x)> ( x ) );
}
inline void Index_Descend_Sort(vector<double*> x, int* beg, int n, int dim)
{
	std::sort(beg, beg+n, compare_indirect_index_descend <decltype(x)> ( x, dim ) );
}

inline bool covers(const double* cub, double *regUp) {
	static int i;
	for (i=0; i<dimensionMinusOne; i++) {
		if (cub[i] < regUp[i]) {
			return false;
		}
	}
	return true;
}

inline bool partCovers(const double* cub, double *regLow) {
	static int i;
	for (i=0; i<dimensionMinusOne; i++) {
		if (cub[i] <= regLow[i]) {
			return false;
		}
	}
	return true;
}

inline int containsBoundary(const double* cub, const double regUp[], const int split, const vector<int> &order) {
	static int j;
	for (j=0; j<split; j++) { // check boundaries
		if (regUp[order[j]] > cub[order[j]]) {
			// boundary contained in region
			return 1;
		}
	}
	// no boundary contained in region
	return 0;
}

inline int isPile(const double* cub, double* regUp) {
	static int pile;
	static int k;

	pile = dimension;
	// check all dimensions of the node
	for (k = 0; k < dimensionMinusOne; k++) {
		// k-boundary of the node's region contained in the cuboid?
		if (cub[k] < regUp[k]) {
			if (pile != dimension) {
				// second dimension occured that is not completely covered
				// ==> cuboid is no pile
				return -1;
			}
			pile = k;
		}
	}
	// if pile == this.dimension then
	// cuboid completely covers region
	// case is not possible since covering cuboids have been removed before

	// region in only one dimenison not completly covered
	// ==> cuboid is a pile
	return pile;
}

inline bool isQuasiPile(const double* cub, const double* regUp, vector<int>& dims, int& p, int& q) {
	static int pile;
	static int k;

	p = dimension;
	q = dimension;
	// check all dimensions of the node
	for (k = 0; k < dimensionMinusOne; k++) {
		// k-boundary of the node's region contained in the cuboid?
		if (cub[dims[k]] < regUp[dims[k]]) {
			if (p != dimension) {
				if (q != dimension) {
					// second dimension occured that is not completely covered
					// ==> cuboid is no pile

					return false;
				}
				q = k;
			}
			else {
				p = k;
			}
		}
	}
	// if pile == this.dimension then
	// cuboid completely covers region
	// case is not possible since covering cuboids have been removed before

	// region in only one dimenison not completly covered
	// ==> cuboid is a pile
	if (q != dimension) {
		return true;
	}
	else
	{
		return false;
	}

}

inline void computeLeafNode(BFTreeNode_largeK *node, const int start) {

	int i, j, k, p, id = -1;

	// updates before the first covering box is inserted
	for (j=start; j<node->partialCoverNumber && !node->isFirstCoverBoxInserted; j++) {
		// skip deleted points, this will not increase the complexity 
		// because the complexity is computed assuming that all boxes will be inserted
		// TODO: consider a clever method of updating node->partialCoverIndex to avoid these checks
		while (j<node->partialCoverNumber && deleted[node->partialCoverIndex[j]]) {
			j++;
		}
		if (j == node->partialCoverNumber) {
			break;
		}
		id = node->partialCoverIndex[j];
		// check if a fully covering box is inserted before id
		if (node->firstCoverBox < id) {
			node->backupInsertCount = j;
			j--;
			id = node->firstCoverBox;
			node->isFirstCoverBoxInserted = true;
			// backup the state before the first covering box is inserted.
			// only contribution of the dominating pile in each dimension was affected by the first fully covering box,
			// after the first covering box is inserted, all the piles have zero contribution,
			// so we do not need to backup contributions of boxes inserted after the fully covering box.
			for (i=0; i<node->usefulDims; i++) {
				node->backupA0[i] = node->A0[i];
				node->backupA1[i] = node->A1[i];
			}
			// cumulate contributions for the non-dominated boxes (if exist) of the 1st, 2nd, ..., (dimensionMinusOne)-th dimensions
			// they cannot be fully covering boxes because the first covering box has not been inserted
			// minusOneContributions[i] maintains the (d-1)-D contribution for the largest i-pile
			for (i=0; i<node->usefulDims; i++) {
				if (node->A0[i] != popsize) {
					contributions[node->partialCoverIndex[node->A0[i]]] += minusOneContributions[i] * (node->xLd - population[id][dimensionMinusOne]);
					node->nodeContributions[node->A0[i]] += minusOneContributions[i] * (node->xLd - population[id][dimensionMinusOne]);
				}
			}
			// first covering box is inserted, we use minusOneContributions[0] to maintain its (d-1)-D contribution
			// node->A1 is useless thereafter
			minusOneContributions[0] = node->uselessDimContribution;
			for (k=0; k<node->usefulDims; k++) {
				if (node->A0[k] != popsize) {
					minusOneContributions[0] *= node->upperbound[node->dims[k]] - population[node->partialCoverIndex[node->A0[k]]][node->dims[k]];
				} else {
					minusOneContributions[0] *= node->delta[k];
				}
			}
		} else {
			// update A0 and A1 when population[id] is a p-pile.
			// also update minusOneContributions accordingly
			for (p=0; p<node->usefulDims; p++) {
				if (population[id][node->dims[p]] < node->upperbound[node->dims[p]]) {
					break;
				}
			}
			if (node->A1[p] != popsize && population[id][node->dims[p]] <= population[node->partialCoverIndex[node->A1[p]]][node->dims[p]]) {
				continue;
			}
			for (i=0; i<node->usefulDims; i++) {
				if (node->A0[i] != popsize) {
					contributions[node->partialCoverIndex[node->A0[i]]] += minusOneContributions[i] * (node->xLd - population[id][dimensionMinusOne]);
					node->nodeContributions[node->A0[i]] += minusOneContributions[i] * (node->xLd - population[id][dimensionMinusOne]);
				}
			}
			if (node->A0[p] == popsize) {
				node->A0[p] = j;
				minusOneContributions[p] = node->uselessDimContribution;
				for (k=0; k<node->usefulDims; k++) {
					// compute (d-1)-D contribution for p-pile and update (d-1)-D contributions for other piles, similarly below
					if (node->A0[k] == popsize) {
						minusOneContributions[p] *= node->delta[k];
					} else {
						if (k == p) {
							minusOneContributions[p] *= population[id][node->dims[p]] - node->lowerbound[node->dims[p]];
						} else {
							minusOneContributions[p] *= node->upperbound[node->dims[k]] - population[node->partialCoverIndex[node->A0[k]]][node->dims[k]];
							minusOneContributions[k] *= (node->upperbound[node->dims[p]] - population[id][node->dims[p]]) / node->delta[p];
						}
					}
				}
			} else if (population[id][node->dims[p]] > population[node->partialCoverIndex[node->A0[p]]][node->dims[p]]) {
				for (k=0; k<node->usefulDims; k++) {
					if (node->A0[k] != popsize) {
						if (k == p) {
							if (node->A1[p] == popsize) {
								minusOneContributions[p] *= (population[id][node->dims[p]] - population[node->partialCoverIndex[node->A0[p]]][node->dims[p]]) /
									(population[node->partialCoverIndex[node->A0[p]]][node->dims[p]] - node->lowerbound[node->dims[p]]);
							} else {
								minusOneContributions[p] *= (population[id][node->dims[p]] - population[node->partialCoverIndex[node->A0[p]]][node->dims[p]]) /
									(population[node->partialCoverIndex[node->A0[p]]][node->dims[p]] - population[node->partialCoverIndex[node->A1[p]]][node->dims[p]]);
							}					
						} else {
							minusOneContributions[k] *= (node->upperbound[node->dims[p]] - population[id][node->dims[p]]) /
								(node->upperbound[node->dims[p]] - population[node->partialCoverIndex[node->A0[p]]][node->dims[p]]);
						}
					}
				}
				node->A1[p] = node->A0[p];
				node->A0[p] = j;
			} else {
				if (node->A1[p] == popsize) {
					minusOneContributions[p] *= (population[node->partialCoverIndex[node->A0[p]]][node->dims[p]] - population[id][node->dims[p]]) /
						(population[node->partialCoverIndex[node->A0[p]]][node->dims[p]] - node->lowerbound[node->dims[p]]);
					node->A1[p] = j;
				} else {
					minusOneContributions[p] *= (population[node->partialCoverIndex[node->A0[p]]][node->dims[p]] - population[id][node->dims[p]]) /
						(population[node->partialCoverIndex[node->A0[p]]][node->dims[p]] - population[node->partialCoverIndex[node->A1[p]]][node->dims[p]]);
					node->A1[p] = j;
				}
			}
		}
		node->xLd = population[id][dimensionMinusOne];		
	} 
	if (node->isFirstCoverBoxInserted) {
		// updates after the first covering box is inserted
		// only the first covering box has nonzero contribution
		double firstCoverContribution = 0.;
		for (; j<node->partialCoverNumber; j++) {
			while (j<node->partialCoverNumber && deleted[node->partialCoverIndex[j]]) {
				j++;
			}
			if (j == node->partialCoverNumber) {
				break;
			}
			id = node->partialCoverIndex[j];
			if (node->secondCoverBox < id) {
				// second covering box is inserted before one partially covering box
				// we sweep from xLd to the second covering box
				id = node->secondCoverBox;
				firstCoverContribution += minusOneContributions[0] * (node->xLd - population[id][dimensionMinusOne]);
				node->xLd = population[id][dimensionMinusOne];
				break;
			}
			for (p=0; p<node->usefulDims; p++) {
				if (population[id][node->dims[p]] < node->upperbound[node->dims[p]]) {
					break;
				}
			}
			if (node->A0[p] != popsize && population[id][node->dims[p]] <= population[node->partialCoverIndex[node->A0[p]]][node->dims[p]]) {
				continue;
			}
			firstCoverContribution += minusOneContributions[0] * (node->xLd - population[id][dimensionMinusOne]);
			if (node->A0[p] == popsize) {
				minusOneContributions[0] *= (node->upperbound[node->dims[p]] - population[id][node->dims[p]]) / node->delta[p];
				node->A0[p] = j;
			} else {
				minusOneContributions[0] *= (node->upperbound[node->dims[p]] - population[id][node->dims[p]]) / (node->upperbound[node->dims[p]] - population[node->partialCoverIndex[node->A0[p]]][node->dims[p]]);
				node->A0[p] = j;
			}
			node->xLd = population[id][dimensionMinusOne];
		}
		node->insertCount = j;
		if (id != node->secondCoverBox) {
			// second covering box does not exist or it is inserted after all the partially covering boxes
			// we sweep from xLd to 0 or to the second covering box
			if (node->secondCoverBox != popsize) {
				firstCoverContribution += minusOneContributions[0] * (node->xLd - population[node->secondCoverBox][dimensionMinusOne]); 
				node->xLd = population[node->secondCoverBox][dimensionMinusOne];
			} else {
				firstCoverContribution += minusOneContributions[0] * node->xLd;
			}
		}
		contributions[node->firstCoverBox] += firstCoverContribution;
		node->firstCoverContribution += firstCoverContribution;
	} else {
		node->insertCount = j;
		if (node->firstCoverBox != popsize) {
			// first covering box is inserted after all the partially covering box
			// actually, we will have j == node->partialCoverNumber in this case
			// we do not need to backup node->A0, node->A1? (TODO)
			node->isFirstCoverBoxInserted = true;
			node->backupInsertCount = j;
			for (i=0; i<node->usefulDims; i++) {
				node->backupA0[i] = node->A0[i];
				node->backupA1[i] = node->A1[i];
			}
			// HVC cumulation for piles (sweep from xLd to the first covering box)
			for (i=0; i<node->usefulDims; i++) {
				if (node->A0[i] != popsize) {
					contributions[node->partialCoverIndex[node->A0[i]]] += minusOneContributions[i] * (node->xLd - population[node->firstCoverBox][dimensionMinusOne]);
					node->nodeContributions[node->A0[i]] += minusOneContributions[i] * (node->xLd - population[node->firstCoverBox][dimensionMinusOne]);
				}
			}
			// HVC cumulation for the first covering box (sweep from the first covering box to 0 or to the second covering box (if exists))
			if (node->secondCoverBox == popsize) {
				node->firstCoverContribution = population[node->firstCoverBox][dimensionMinusOne] * node->uselessDimContribution;
				node->xLd = population[node->firstCoverBox][dimensionMinusOne];
			} else {
				node->firstCoverContribution = (population[node->firstCoverBox][dimensionMinusOne] - population[node->secondCoverBox][dimensionMinusOne]) * node->uselessDimContribution;
				node->xLd = population[node->secondCoverBox][dimensionMinusOne];
			}
			for (k=0; k<node->usefulDims; k++) {
				if (node->A0[k] == popsize) {
					node->firstCoverContribution *= node->delta[k];
				} else {
					node->firstCoverContribution *= node->upperbound[node->dims[k]] - population[node->partialCoverIndex[node->A0[k]]][node->dims[k]];
				}
			}
			contributions[node->firstCoverBox] += node->firstCoverContribution;
		} else {
			// in this case, there is no covering box, and all the partially covering boxes have been visited,
			// so we directly sweep from xLd to 0
			for (i=0; i<node->usefulDims; i++) {
				if (node->A0[i] != popsize) {
					contributions[node->partialCoverIndex[node->A0[i]]] += minusOneContributions[i] * node->xLd;
					node->nodeContributions[node->A0[i]] += minusOneContributions[i] * node->xLd;
				}
			}
		}
	}
}

inline void computeLeafNode(BFTreeNode_smallK *node, const int start) {

	int i, j, k, p, id = -1;

	// updates before the first covering box is inserted
	for (j=start; j<node->partialCoverNumber && !node->isFirstCoverBoxInserted; j++) {
		// skip deleted points, this will not increase the complexity 
		// because the complexity is computed assuming that all boxes will be inserted
		// TODO: consider a clever method of updating node->partialCoverIndex to avoid these checks
		while (j<node->partialCoverNumber && deleted[node->partialCoverIndex[j]]) {
			j++;
		}
		if (j == node->partialCoverNumber) {
			break;
		}
		id = node->partialCoverIndex[j];
		// check if a fully covering box is inserted before id
		if (node->firstCoverBox < id) {
			node->backupInsertCount = j;
			j--;
			id = node->firstCoverBox;
			node->isFirstCoverBoxInserted = true;
			// backup the state before the first covering box is inserted.
			// only contribution of the dominating pile in each dimension was affected by the first fully covering box,
			// after the first covering box is inserted, all the piles have zero contribution,
			// so we do not need to backup contributions of boxes inserted after the fully covering box.
			for (i=0; i<node->usefulDims; i++) {
				node->backupA0[i] = node->A0[i];
				node->backupA1[i] = node->A1[i];
			}
			// cumulate contributions for the non-dominated boxes (if exist) of the 1st, 2nd, ..., (dimensionMinusOne)-th dimensions
			// they cannot be fully covering boxes because the first covering box has not been inserted
			// minusOneContributions[i] maintains the (d-1)-D contribution for the largest i-pile
			for (i=0; i<node->usefulDims; i++) {
				if (node->A0[i] != popsize) {
					contributions[node->partialCoverIndex[node->A0[i]]] += minusOneContributions[i] * (node->xLd - population[id][dimensionMinusOne]);
					node->nodeContributions[node->A0[i]] += minusOneContributions[i] * (node->xLd - population[id][dimensionMinusOne]);
				}
			}
			// first covering box is inserted, we use minusOneContributions[0] to maintain its (d-1)-D contribution
			// node->A1 is useless thereafter
			minusOneContributions[0] = node->uselessDimContribution;
			for (k=0; k<node->usefulDims; k++) {
				if (node->A0[k] != popsize) {
					minusOneContributions[0] *= node->upperbound[node->dims[k]] - population[node->partialCoverIndex[node->A0[k]]][node->dims[k]];
				} else {
					minusOneContributions[0] *= node->delta[k];
				}
			}
		} else {
			// update A0 and A1 when population[id] is a p-pile.
			// also update minusOneContributions accordingly
			for (p=0; p<node->usefulDims; p++) {
				if (population[id][node->dims[p]] < node->upperbound[node->dims[p]]) {
					break;
				}
			}
			if (node->A1[p] != popsize && population[id][node->dims[p]] <= population[node->partialCoverIndex[node->A1[p]]][node->dims[p]]) {
				continue;
			}
			for (i=0; i<node->usefulDims; i++) {
				if (node->A0[i] != popsize) {
					contributions[node->partialCoverIndex[node->A0[i]]] += minusOneContributions[i] * (node->xLd - population[id][dimensionMinusOne]);
					node->nodeContributions[node->A0[i]] += minusOneContributions[i] * (node->xLd - population[id][dimensionMinusOne]);
				}
			}
			if (node->A0[p] == popsize) {
				node->A0[p] = j;
				minusOneContributions[p] = node->uselessDimContribution;
				for (k=0; k<node->usefulDims; k++) {
					// compute (d-1)-D contribution for p-pile and update (d-1)-D contributions for other piles, similarly below
					if (node->A0[k] == popsize) {
						minusOneContributions[p] *= node->delta[k];
					} else {
						if (k == p) {
							minusOneContributions[p] *= population[id][node->dims[p]] - node->lowerbound[node->dims[p]];
						} else {
							minusOneContributions[p] *= node->upperbound[node->dims[k]] - population[node->partialCoverIndex[node->A0[k]]][node->dims[k]];
							minusOneContributions[k] *= (node->upperbound[node->dims[p]] - population[id][node->dims[p]]) / node->delta[p];
						}
					}
				}
			} else if (population[id][node->dims[p]] > population[node->partialCoverIndex[node->A0[p]]][node->dims[p]]) {
				for (k=0; k<node->usefulDims; k++) {
					if (node->A0[k] != popsize) {
						if (k == p) {
							if (node->A1[p] == popsize) {
								minusOneContributions[p] *= (population[id][node->dims[p]] - population[node->partialCoverIndex[node->A0[p]]][node->dims[p]]) /
									(population[node->partialCoverIndex[node->A0[p]]][node->dims[p]] - node->lowerbound[node->dims[p]]);
							} else {
								minusOneContributions[p] *= (population[id][node->dims[p]] - population[node->partialCoverIndex[node->A0[p]]][node->dims[p]]) /
									(population[node->partialCoverIndex[node->A0[p]]][node->dims[p]] - population[node->partialCoverIndex[node->A1[p]]][node->dims[p]]);
							}					
						} else {
							minusOneContributions[k] *= (node->upperbound[node->dims[p]] - population[id][node->dims[p]]) /
								(node->upperbound[node->dims[p]] - population[node->partialCoverIndex[node->A0[p]]][node->dims[p]]);
						}
					}
				}
				node->A1[p] = node->A0[p];
				node->A0[p] = j;
			} else {
				if (node->A1[p] == popsize) {
					minusOneContributions[p] *= (population[node->partialCoverIndex[node->A0[p]]][node->dims[p]] - population[id][node->dims[p]]) /
						(population[node->partialCoverIndex[node->A0[p]]][node->dims[p]] - node->lowerbound[node->dims[p]]);
					node->A1[p] = j;
				} else {
					minusOneContributions[p] *= (population[node->partialCoverIndex[node->A0[p]]][node->dims[p]] - population[id][node->dims[p]]) /
						(population[node->partialCoverIndex[node->A0[p]]][node->dims[p]] - population[node->partialCoverIndex[node->A1[p]]][node->dims[p]]);
					node->A1[p] = j;
				}
			}
		}
		node->xLd = population[id][dimensionMinusOne];		
	} 
	if (node->isFirstCoverBoxInserted) {
		// updates after the first covering box is inserted
		// only the first covering box has nonzero contribution
		double firstCoverContribution = 0.;
		for (; j<node->partialCoverNumber; j++) {
			while (j<node->partialCoverNumber && deleted[node->partialCoverIndex[j]]) {
				j++;
			}
			if (j == node->partialCoverNumber) {
				break;
			}
			id = node->partialCoverIndex[j];
			if (node->secondCoverBox < id) {
				// second covering box is inserted before one partially covering box
				// we sweep from xLd to the second covering box
				id = node->secondCoverBox;
				firstCoverContribution += minusOneContributions[0] * (node->xLd - population[id][dimensionMinusOne]);
				node->xLd = population[id][dimensionMinusOne];
				break;
			}
			for (p=0; p<node->usefulDims; p++) {
				if (population[id][node->dims[p]] < node->upperbound[node->dims[p]]) {
					break;
				}
			}
			if (node->A0[p] != popsize && population[id][node->dims[p]] <= population[node->partialCoverIndex[node->A0[p]]][node->dims[p]]) {
				continue;
			}
			firstCoverContribution += minusOneContributions[0] * (node->xLd - population[id][dimensionMinusOne]);
			if (node->A0[p] == popsize) {
				minusOneContributions[0] *= (node->upperbound[node->dims[p]] - population[id][node->dims[p]]) / node->delta[p];
				node->A0[p] = j;
			} else {
				minusOneContributions[0] *= (node->upperbound[node->dims[p]] - population[id][node->dims[p]]) / (node->upperbound[node->dims[p]] - population[node->partialCoverIndex[node->A0[p]]][node->dims[p]]);
				node->A0[p] = j;
			}
			node->xLd = population[id][dimensionMinusOne];
		}
		node->insertCount = j;
		if (id != node->secondCoverBox) {
			// second covering box does not exist or it is inserted after all the partially covering boxes
			// we sweep from xLd to 0 or to the second covering box
			if (node->secondCoverBox != popsize) {
				firstCoverContribution += minusOneContributions[0] * (node->xLd - population[node->secondCoverBox][dimensionMinusOne]); 
				node->xLd = population[node->secondCoverBox][dimensionMinusOne];
			} else {
				firstCoverContribution += minusOneContributions[0] * node->xLd;
			}
		}
		contributions[node->firstCoverBox] += firstCoverContribution;
		node->firstCoverContribution += firstCoverContribution;
	} else {
		node->insertCount = j;
		if (node->firstCoverBox != popsize) {
			// first covering box is inserted after all the partially covering box
			// actually, we will have j == node->partialCoverNumber in this case
			// we do not need to backup node->A0, node->A1? (TODO)
			node->isFirstCoverBoxInserted = true;
			node->backupInsertCount = j;
			for (i=0; i<node->usefulDims; i++) {
				node->backupA0[i] = node->A0[i];
				node->backupA1[i] = node->A1[i];
			}
			// HVC cumulation for piles (sweep from xLd to the first covering box)
			for (i=0; i<node->usefulDims; i++) {
				if (node->A0[i] != popsize) {
					contributions[node->partialCoverIndex[node->A0[i]]] += minusOneContributions[i] * (node->xLd - population[node->firstCoverBox][dimensionMinusOne]);
					node->nodeContributions[node->A0[i]] += minusOneContributions[i] * (node->xLd - population[node->firstCoverBox][dimensionMinusOne]);
				}
			}
			// HVC cumulation for the first covering box (sweep from the first covering box to 0 or to the second covering box (if exists))
			if (node->secondCoverBox == popsize) {
				node->firstCoverContribution = population[node->firstCoverBox][dimensionMinusOne] * node->uselessDimContribution;
				node->xLd = population[node->firstCoverBox][dimensionMinusOne];
			} else {
				node->firstCoverContribution = (population[node->firstCoverBox][dimensionMinusOne] - population[node->secondCoverBox][dimensionMinusOne]) * node->uselessDimContribution;
				node->xLd = population[node->secondCoverBox][dimensionMinusOne];
			}
			for (k=0; k<node->usefulDims; k++) {
				if (node->A0[k] == popsize) {
					node->firstCoverContribution *= node->delta[k];
				} else {
					node->firstCoverContribution *= node->upperbound[node->dims[k]] - population[node->partialCoverIndex[node->A0[k]]][node->dims[k]];
				}
			}
			contributions[node->firstCoverBox] += node->firstCoverContribution;
		} else {
			// in this case, there is no covering box, and all the partially covering boxes have been visited,
			// so we directly sweep from xLd to 0
			for (i=0; i<node->usefulDims; i++) {
				if (node->A0[i] != popsize) {
					contributions[node->partialCoverIndex[node->A0[i]]] += minusOneContributions[i] * node->xLd;
					node->nodeContributions[node->A0[i]] += minusOneContributions[i] * node->xLd;
				}
			}
		}
	}
}

inline void buildTree(BFTreeNode_largeK *node) {

	unsigned int i, j;

	int id, iterCount, count = 0, nonPiles = 0;
	for (i=0; i<dimensionMinusOne; i++) {
		dimensionPileCount[i] = 0;
	}
	int pp, qq, rr = -1, ss;
	bool fullQuasiPile = true;
	for (i=0; i<node->partialCoverNumber; i++) {
		// cut off tree
		// since k (i.e., reservedNumber) is large, this check can be easily satisfied
		if (node->dominatedCount[i] + node->fullyCount > popsize - reservedNumber) {
			break;
		}
		id = node->partialCoverIndex[i];
		if (covers(population[id], node->upperbound)) {
			// save first two fully covering boxes (considering father nodes)
			if (id < node->firstCoverBox) {
				node->secondCoverBox = node->firstCoverBox;
				node->firstCoverBox = id;
			} else if (id < node->secondCoverBox) {
				node->secondCoverBox = id;
			}
			// save all fully covering box of this node
			node->fullyCoverBoxes[node->fullyCount] = id;
			node->fullyCount++;
		}
		else {
			piles[id] = isPile(population[id], node->upperbound);
			if (piles[id] == -1) {
				nonPiles++;
				// check whether all the non-piles become quasipiles in the same dimensions
				if (!node->isQuasiPile && fullQuasiPile) {
					if (isQuasiPile(population[id], node->upperbound, node->dims, pp, qq)) {
						if (rr == -1) {
							rr = pp;
							ss = qq;
						} else if (pp != rr || qq != ss) {
							fullQuasiPile = false;
						}
					} else {
						fullQuasiPile = false;
					}
				}
			} else {
				dimensionPileCount[piles[id]]++;
			} 
			node->partialCoverIndex[count] = id;
			node->dominatedCount[count] += node->fullyCount;
			count++;
		}
	}

	int backupi = i;

	// compute leaf node
	if (nonPiles == 0 && count <= dSqrtDataNumber * dimension * dimensionMinusOne / 2) {
		node->dims.clear();
		node->uselessDimContribution = 1.;
		for (i=0; i<dimensionMinusOne; i++) {
			if (dimensionPileCount[i] > 0) {
				node->dims.push_back(i);
			} else {
				node->uselessDimContribution *= node->upperbound[i] - node->lowerbound[i];
			}
		}
		node->usefulDims = node->dims.size();
		node->isFirstCoverBoxInserted = false;
		node->indexOfMyFirstCoverBox = 0;
		node->indexOfMySecondCoverBox = 1;
		node->partialCoverNumber = count;
		node->firstCoverContribution = 0.;
		node->A0 = new int[node->usefulDims];
		node->A1 = new int[node->usefulDims];
		node->backupA0 = new int[node->usefulDims];
		node->backupA1 = new int[node->usefulDims];
		node->delta = new double[node->usefulDims];
		node->nodeContributions = new double[count];
		for (i=0; i<node->usefulDims; i++) {
			node->A0[i] = popsize;	// A^i_1 in the paper, not equal to any index in partialCoverIndex before the computation, i.e., it is undefined
			node->A1[i] = popsize;	// A^i_2 in the paper
			node->delta[i] = node->upperbound[node->dims[i]] - node->lowerbound[node->dims[i]];
		}
		for (i=0; i<count; i++) {
			node->nodeContributions[i] = 0.;
		}
		computeLeafNode(node, 0);

		for (i=0; i<dimensionMinusOne; i++) {
			delete [] node->LinkedList[i];
		}
		delete [] node->LinkedListSize;
		delete [] node->dominatedCount;
		return;
	}

	// update LinkedList since the tree is cut off
	if (backupi < node->partialCoverNumber) {
		for (i=0; i<dimensionMinusOne; i++) {
			iterCount = 0;
			for (j=0; j<node->LinkedListSize[i]; j++) {
				// count <> 0, otherwise, no point would partially cover this node, then this node would be a leaf node
				if (node->LinkedList[i][j] <= node->partialCoverIndex[count - 1]) {
					node->LinkedList[i][iterCount] = node->LinkedList[i][j];
					iterCount++;
				}
			}
			node->LinkedListSize[i] = iterCount;
		}
	}

	if (!node->isQuasiPile && nonPiles > 0 && nonPiles < dSqrtDataNumber && fullQuasiPile) {
		// in this case, we use the space partition strategy in the HVC4D-GS algorithm
		node->quasiPiles = vector<int>(2);
		node->quasiPiles[0] = pp;
		node->quasiPiles[1] = qq;
		node->isQuasiPile = true;
	}

	// find which dimension and which coordinate to split
	int split, middleIndex;
	bool flag;

	if (nonPiles == 0) {
		// when there are all piles, directly split along the dimension which has the most piles
		flag = true;
		split = 0;
		for (i=0; i<dimensionMinusOne; i++) {
			if (node->LinkedListSize[i] > node->LinkedListSize[split]) {
				split = i;
			}
		}
		middleIndex = node->LinkedListSize[split] / 2;
	} else {
		if (node->isQuasiPile) {
			// the space partition strategy in the HVC4D-GS algorithm
			flag = true;
			int nonPileCount;
			static vector<int> Ids;
			Ids = vector<int>(2);	
			for (i=0; i<2; i++) {
				nonPileCount = 0;
				for (j=0; j<count; j++) {
					if (piles[node->LinkedList[node->quasiPiles[i]][j]] == -1) {
						nonPileCount++;
						if (nonPileCount>=nonPiles/2) {
							Ids[i] = j;
							break;
						}
					}
				}
			}
			int select;
			if (Ids[0] == Ids[1]) {
				select = alter;
				alter = 1 - alter;
			} else if (Ids[0] > Ids[1]) {
				select = 0;
			} else {
				select = 1;
			}
			split = node->quasiPiles[select];
			middleIndex = Ids[select];
		} else {
			// the original space partition method in HOY method
			flag = false;
			split = node->split;
			middleIndex = -1;
			int boundSize = 0, noBoundSize = 0;

			do {
				for (i=0; i<node->LinkedListSize[split]; i++) {
					int contained = containsBoundary(population[node->LinkedList[split][i]], node->upperbound, split, node->dims);
					if (contained == 1) {
						boundaries[boundSize] = i;
						boundSize++;
					} else {
						noBoundaries[noBoundSize] = i;
						noBoundSize++;
					}
				}

				if (boundSize > 0) {
					middleIndex = boundaries[boundSize / 2];
				} else if (noBoundSize > dSqrtDataNumber) {
					middleIndex = noBoundaries[noBoundSize / 2];
				} else {
					split++;
					noBoundSize = 0;
				}
			} while (middleIndex == -1);
		}
	}
	int trueDimension = node->dims[split];

	// data for left child
	BFTreeNode_largeK *leftChild = new BFTreeNode_largeK();
	leftChild->partialCoverIndex = new int[count];
	leftChild->dominatedCount = new int[count];
	leftChild->fullyCoverBoxes = new int[count];
	leftChild->fullyCount = 0;
	leftChild->isQuasiPile = node->isQuasiPile;
	leftChild->partialCoverNumber = count;
	leftChild->firstCoverBox = node->firstCoverBox;
	leftChild->secondCoverBox = node->secondCoverBox;
	leftChild->lowerbound = new double[dimensionMinusOne];
	leftChild->upperbound = new double[dimensionMinusOne];
	for (i=0; i<dimensionMinusOne; i++) {
		leftChild->lowerbound[i] = node->lowerbound[i];
		leftChild->upperbound[i] = node->upperbound[i];
	}
	leftChild->upperbound[trueDimension] = population[node->LinkedList[split][middleIndex]][trueDimension];
	// prepare LinkedList for left child
	// note that, in each dimension, we only record points which partially cover (but not cover) the interval of the node
	leftChild->LinkedList = vector<int*>(dimensionMinusOne);
	leftChild->LinkedListSize = new int[dimensionMinusOne];
	leftChild->LinkedList[split] = new int[middleIndex];
	for (i=0; i<middleIndex; i++) {
		if (population[node->LinkedList[split][i]][trueDimension] < leftChild->upperbound[trueDimension]) {
			leftChild->LinkedList[split][i] = node->LinkedList[split][i];
		} else {
			break;
		}
	}
	leftChild->LinkedListSize[split] = i;
	for (i=0; i<dimensionMinusOne; i++) {
		if (i == split) {
			continue;
		}
		leftChild->LinkedList[i] = new int[min(count, node->LinkedListSize[i])];
		iterCount = 0;
		for (j=0; j<node->LinkedListSize[i]; j++) {
			if (population[node->LinkedList[i][j]][node->dims[i]] < leftChild->upperbound[node->dims[i]]) {
				leftChild->LinkedList[i][iterCount] = node->LinkedList[i][j];
				iterCount++;
			}
		}
		leftChild->LinkedListSize[i] = iterCount;
	}
	if (flag) {
		leftChild->dims = node->dims;
		leftChild->quasiPiles = node->quasiPiles;
	} else {
		// reorder the dimension to split
		// only for the original space partition method in the HOY method
		leftChild->split = split;
		leftChild->dims = vector<int>(dimensionMinusOne);
		for (i=0; i<split; i++) {
			leftChild->dims[i] = node->dims[i];
		}
		int *newList = leftChild->LinkedList[split];
		int newListSize = leftChild->LinkedListSize[split];
		for (i=split; i<dimensionMinusOne - 1; i++) {
			leftChild->dims[i] = node->dims[i + 1];
			leftChild->LinkedList[i] = leftChild->LinkedList[i + 1];
			leftChild->LinkedListSize[i] = leftChild->LinkedListSize[i + 1];
		}
		leftChild->dims[dimensionMinusOne - 1] = node->dims[split];
		leftChild->LinkedList[dimensionMinusOne - 1] = newList;
		leftChild->LinkedListSize[dimensionMinusOne - 1] = newListSize;
	}

	// similar codes for right child
	BFTreeNode_largeK *rightChild = new BFTreeNode_largeK();
	rightChild->partialCoverIndex = new int[count];
	rightChild->dominatedCount = new int[count];
	int count2 = 0;
	for (i=0; i<count; i++) {
		id = node->partialCoverIndex[i];
		leftChild->partialCoverIndex[i] = id;
		leftChild->dominatedCount[i] = node->dominatedCount[i];
		if (population[id][trueDimension] > leftChild->upperbound[trueDimension]) {
			rightChild->partialCoverIndex[count2] = id;
			rightChild->dominatedCount[count2] = node->dominatedCount[i];
			count2++;
		} 
	}
	if (count2 > 0 || node->firstCoverBox != popsize) {
		rightChild->isQuasiPile = node->isQuasiPile;
		rightChild->dims = leftChild->dims;
		rightChild->partialCoverNumber = count2;
		rightChild->firstCoverBox = node->firstCoverBox;
		rightChild->secondCoverBox = node->secondCoverBox;
		rightChild->fullyCoverBoxes = new int[count2];
		rightChild->fullyCount = 0;
		rightChild->lowerbound = new double[dimensionMinusOne];
		rightChild->upperbound = new double[dimensionMinusOne];
		for (i=0; i<dimensionMinusOne; i++) {
			rightChild->lowerbound[i] = node->lowerbound[i];
			rightChild->upperbound[i] = node->upperbound[i];
		}
		rightChild->lowerbound[trueDimension] = leftChild->upperbound[trueDimension];

		rightChild->LinkedList = vector<int*>(dimensionMinusOne);
		rightChild->LinkedListSize = new int[dimensionMinusOne];
		rightChild->LinkedList[split] = new int[node->LinkedListSize[split] - middleIndex];
		for (i=middleIndex + 1; i<node->LinkedListSize[split]; i++) {
			rightChild->LinkedList[split][i - middleIndex - 1] = node->LinkedList[split][i];
		}
		rightChild->LinkedListSize[split] = node->LinkedListSize[split] - middleIndex - 1;
		for (i=0; i<rightChild->LinkedListSize[split]; i++) {
			if (population[rightChild->LinkedList[split][i]][trueDimension] > rightChild->lowerbound[trueDimension]) {
				break;
			}
		}
		if (i > 0) {
			for (j=i; j<rightChild->LinkedListSize[split]; j++) {
				rightChild->LinkedList[split][j - i] = rightChild->LinkedList[split][j];
			}
			rightChild->LinkedListSize[split] -= i;
		}
		for (i=0; i<dimensionMinusOne; i++) {
			if (i == split) {
				continue;
			}
			rightChild->LinkedList[i] = new int[min(count2, node->LinkedListSize[i])];
			iterCount = 0;
			for (j=0; j<node->LinkedListSize[i]; j++) {
				if (population[node->LinkedList[i][j]][trueDimension] > leftChild->upperbound[trueDimension] && 
					population[node->LinkedList[i][j]][node->dims[i]] > rightChild->lowerbound[node->dims[i]]) 
				{
					rightChild->LinkedList[i][iterCount] = node->LinkedList[i][j];
					iterCount++;
				}
			}
			rightChild->LinkedListSize[i] = iterCount;
		}
		if (flag) {
			rightChild->quasiPiles = node->quasiPiles;
		} else {
			rightChild->split = split;
			int *newList = rightChild->LinkedList[split];
			int newListSize = rightChild->LinkedListSize[split];
			for (i=split; i<dimensionMinusOne - 1; i++) {
				rightChild->LinkedList[i] = rightChild->LinkedList[i + 1];
				rightChild->LinkedListSize[i] = rightChild->LinkedListSize[i + 1];
			}
			rightChild->LinkedList[dimensionMinusOne - 1] = newList;
			rightChild->LinkedListSize[dimensionMinusOne - 1] = newListSize;
		}
	} else {
		delete [] rightChild->partialCoverIndex;
		delete [] rightChild->dominatedCount;
		delete rightChild;
		rightChild = nullptr;
	}
	for (i=0; i<dimensionMinusOne; i++) {
		delete [] node->LinkedList[i];
	}
	delete [] node->LinkedListSize;
	delete [] node->partialCoverIndex;
	delete [] node->dominatedCount;

	// recursive call for children
	buildTree(leftChild);
	if (rightChild != nullptr) {
		buildTree(rightChild);
	}
	node->leftChild = leftChild;
	node->rightChild = rightChild;	
	node->indexOfMyFirstCoverBox = 0;
	node->indexOfMySecondCoverBox = 1;
}

inline void buildTree(BFTreeNode_smallK *node) {

	unsigned int i, j;

	int id, iterCount, count = 0, nonPiles = 0;
	for (i=0; i<dimensionMinusOne; i++) {
		dimensionPileCount[i] = 0;
	}
	int pp, qq, rr = -1, ss;
	bool fullQuasiPile = true;
	for (i=0; i<node->partialCoverNumber; i++) {
		id = node->partialCoverIndex[i];
		if (covers(population[id], node->upperbound)) {
			// save first two fully covering boxes (considering father nodes)
			if (id < node->firstCoverBox) {
				node->secondCoverBox = node->firstCoverBox;
				node->firstCoverBox = id;
			} else if (id < node->secondCoverBox) {
				node->secondCoverBox = id;
			}
			// save all fully covering box of this node
			node->fullyCoverBoxes[node->fullyCount] = id;
			node->fullyCount++;
		}
		else {
			piles[id] = isPile(population[id], node->upperbound);
			if (piles[id] == -1) {
				nonPiles++;
				// check whether all the non-piles become quasipiles in the same dimensions
				if (!node->isQuasiPile && fullQuasiPile) {
					if (isQuasiPile(population[id], node->upperbound, node->dims, pp, qq)) {
						if (rr == -1) {
							rr = pp;
							ss = qq;
						} else if (pp != rr || qq != ss) {
							fullQuasiPile = false;
						}
					} else {
						fullQuasiPile = false;
					}
				}
			} else {
				dimensionPileCount[piles[id]]++;
			} 
			node->partialCoverIndex[count] = id;
			count++;
		}
	}

	// compute leaf node
	if (nonPiles == 0 && count <= dSqrtDataNumber * dimension * dimensionMinusOne / 2) {
		node->dims.clear();
		node->uselessDimContribution = 1.;
		for (i=0; i<dimensionMinusOne; i++) {
			if (dimensionPileCount[i] > 0) {
				node->dims.push_back(i);
			} else {
				node->uselessDimContribution *= node->upperbound[i] - node->lowerbound[i];
			}
		}
		node->usefulDims = node->dims.size();
		node->isFirstCoverBoxInserted = false;
		node->indexOfMyFirstCoverBox = 0;
		node->indexOfMySecondCoverBox = 1;
		node->partialCoverNumber = count;
		node->firstCoverContribution = 0.;
		node->A0 = new int[node->usefulDims];
		node->A1 = new int[node->usefulDims];
		node->backupA0 = new int[node->usefulDims];
		node->backupA1 = new int[node->usefulDims];
		node->delta = new double[node->usefulDims];
		node->nodeContributions = new double[count];
		for (i=0; i<node->usefulDims; i++) {
			node->A0[i] = popsize;	// A^i_1 in the paper, not equal to any index in partialCoverIndex before the computation, i.e., it is undefined
			node->A1[i] = popsize;	// A^i_2 in the paper
			node->delta[i] = node->upperbound[node->dims[i]] - node->lowerbound[node->dims[i]];
		}
		for (i=0; i<count; i++) {
			node->nodeContributions[i] = 0.;
		}
		computeLeafNode(node, 0);

		for (i=0; i<dimensionMinusOne; i++) {
			delete [] node->LinkedList[i];
		}
		delete [] node->LinkedListSize;
		return;
	}

	if (!node->isQuasiPile && nonPiles > 0 && nonPiles < dSqrtDataNumber && fullQuasiPile) {
		node->quasiPiles = vector<int>(2);
		node->quasiPiles[0] = pp;
		node->quasiPiles[1] = qq;
		node->isQuasiPile = true;
	}

	// find which dimension and which coordinate to split
	int split, middleIndex;
	bool flag;

	if (nonPiles == 0) {
		// when there are all piles, directly split along the dimension which has the most piles
		flag = true;
		split = 0;
		for (i=0; i<dimensionMinusOne; i++) {
			if (node->LinkedListSize[i] > node->LinkedListSize[split]) {
				split = i;
			}
		}
		middleIndex = node->LinkedListSize[split] / 2;
	} else {
		if (node->isQuasiPile) {
			// the space partition strategy in the HVC4D-GS algorithm
			flag = true;
			int nonPileCount;
			static vector<int> Ids;
			Ids = vector<int>(2);	
			for (i=0; i<2; i++) {
				nonPileCount = 0;
				for (j=0; j<count; j++) {
					if (piles[node->LinkedList[node->quasiPiles[i]][j]] == -1) {
						nonPileCount++;
						if (nonPileCount>=nonPiles/2) {
							Ids[i] = j;
							break;
						}
					}
				}
			}
			int select;
			if (Ids[0] == Ids[1]) {
				select = alter;
				alter = 1 - alter;
			} else if (Ids[0] > Ids[1]) {
				select = 0;
			} else {
				select = 1;
			}
			split = node->quasiPiles[select];
			middleIndex = Ids[select];
		} else {
			// the original space partition method in HOY method
			flag = false;
			split = node->split;
			middleIndex = -1;
			int boundSize = 0, noBoundSize = 0;

			do {
				for (i=0; i<node->LinkedListSize[split]; i++) {
					int contained = containsBoundary(population[node->LinkedList[split][i]], node->upperbound, split, node->dims);
					if (contained == 1) {
						boundaries[boundSize] = i;
						boundSize++;
					} else {
						noBoundaries[noBoundSize] = i;
						noBoundSize++;
					}
				}

				if (boundSize > 0) {
					middleIndex = boundaries[boundSize / 2];
				} else if (noBoundSize > dSqrtDataNumber) {
					middleIndex = noBoundaries[noBoundSize / 2];
				} else {
					split++;
					noBoundSize = 0;
				}
			} while (middleIndex == -1);
		}
	}
	int trueDimension = node->dims[split];

	// data for left child
	BFTreeNode_smallK *leftChild = new BFTreeNode_smallK();
	leftChild->partialCoverIndex = new int[count];
	leftChild->fullyCoverBoxes = new int[count];
	leftChild->fullyCount = 0;
	leftChild->isQuasiPile = node->isQuasiPile;
	leftChild->partialCoverNumber = count;
	leftChild->firstCoverBox = node->firstCoverBox;
	leftChild->secondCoverBox = node->secondCoverBox;
	leftChild->lowerbound = new double[dimensionMinusOne];
	leftChild->upperbound = new double[dimensionMinusOne];
	for (i=0; i<dimensionMinusOne; i++) {
		leftChild->lowerbound[i] = node->lowerbound[i];
		leftChild->upperbound[i] = node->upperbound[i];
	}
	leftChild->upperbound[trueDimension] = population[node->LinkedList[split][middleIndex]][trueDimension];
	// prepare LinkedList for left child
	// note that, in each dimension, we only record points which partially cover (but not cover) the interval of the node
	leftChild->LinkedList = vector<int*>(dimensionMinusOne);
	leftChild->LinkedListSize = new int[dimensionMinusOne];
	leftChild->LinkedList[split] = new int[middleIndex];
	for (i=0; i<middleIndex; i++) {
		if (population[node->LinkedList[split][i]][trueDimension] < leftChild->upperbound[trueDimension]) {
			leftChild->LinkedList[split][i] = node->LinkedList[split][i];
		} else {
			break;
		}
	}
	leftChild->LinkedListSize[split] = i;
	for (i=0; i<dimensionMinusOne; i++) {
		if (i == split) {
			continue;
		}
		leftChild->LinkedList[i] = new int[min(count, node->LinkedListSize[i])];
		iterCount = 0;
		for (j=0; j<node->LinkedListSize[i]; j++) {
			if (population[node->LinkedList[i][j]][node->dims[i]] < leftChild->upperbound[node->dims[i]]) {
				leftChild->LinkedList[i][iterCount] = node->LinkedList[i][j];
				iterCount++;
			}
		}
		leftChild->LinkedListSize[i] = iterCount;
	}
	if (flag) {
		leftChild->dims = node->dims;
		leftChild->quasiPiles = node->quasiPiles;
	} else {
		// reorder the dimension to split
		// only for the original space partition method in the HOY method
		leftChild->split = split;
		leftChild->dims = vector<int>(dimensionMinusOne);
		for (i=0; i<split; i++) {
			leftChild->dims[i] = node->dims[i];
		}
		int *newList = leftChild->LinkedList[split];
		int newListSize = leftChild->LinkedListSize[split];
		for (i=split; i<dimensionMinusOne - 1; i++) {
			leftChild->dims[i] = node->dims[i + 1];
			leftChild->LinkedList[i] = leftChild->LinkedList[i + 1];
			leftChild->LinkedListSize[i] = leftChild->LinkedListSize[i + 1];
		}
		leftChild->dims[dimensionMinusOne - 1] = node->dims[split];
		leftChild->LinkedList[dimensionMinusOne - 1] = newList;
		leftChild->LinkedListSize[dimensionMinusOne - 1] = newListSize;
	}

	// similar codes for right child
	BFTreeNode_smallK *rightChild = new BFTreeNode_smallK();
	rightChild->partialCoverIndex = new int[count];
	int count2 = 0;
	for (i=0; i<count; i++) {
		id = node->partialCoverIndex[i];
		leftChild->partialCoverIndex[i] = id;
		if (population[id][trueDimension] > leftChild->upperbound[trueDimension]) {
			rightChild->partialCoverIndex[count2] = id;
			count2++;
		} 
	}
	if (count2 > 0 || node->firstCoverBox != popsize) {
		rightChild->isQuasiPile = node->isQuasiPile;
		rightChild->dims = leftChild->dims;
		rightChild->partialCoverNumber = count2;
		rightChild->firstCoverBox = node->firstCoverBox;
		rightChild->secondCoverBox = node->secondCoverBox;
		rightChild->fullyCoverBoxes = new int[count2];
		rightChild->fullyCount = 0;
		rightChild->lowerbound = new double[dimensionMinusOne];
		rightChild->upperbound = new double[dimensionMinusOne];
		for (i=0; i<dimensionMinusOne; i++) {
			rightChild->lowerbound[i] = node->lowerbound[i];
			rightChild->upperbound[i] = node->upperbound[i];
		}
		rightChild->lowerbound[trueDimension] = leftChild->upperbound[trueDimension];

		rightChild->LinkedList = vector<int*>(dimensionMinusOne);
		rightChild->LinkedListSize = new int[dimensionMinusOne];
		rightChild->LinkedList[split] = new int[node->LinkedListSize[split] - middleIndex];
		for (i=middleIndex + 1; i<node->LinkedListSize[split]; i++) {
			rightChild->LinkedList[split][i - middleIndex - 1] = node->LinkedList[split][i];
		}
		rightChild->LinkedListSize[split] = node->LinkedListSize[split] - middleIndex - 1;
		for (i=0; i<rightChild->LinkedListSize[split]; i++) {
			if (population[rightChild->LinkedList[split][i]][trueDimension] > rightChild->lowerbound[trueDimension]) {
				break;
			}
		}
		if (i > 0) {
			for (j=i; j<rightChild->LinkedListSize[split]; j++) {
				rightChild->LinkedList[split][j - i] = rightChild->LinkedList[split][j];
			}
			rightChild->LinkedListSize[split] -= i;
		}
		for (i=0; i<dimensionMinusOne; i++) {
			if (i == split) {
				continue;
			}
			rightChild->LinkedList[i] = new int[min(count2, node->LinkedListSize[i])];
			iterCount = 0;
			for (j=0; j<node->LinkedListSize[i]; j++) {
				if (population[node->LinkedList[i][j]][trueDimension] > leftChild->upperbound[trueDimension] && 
					population[node->LinkedList[i][j]][node->dims[i]] > rightChild->lowerbound[node->dims[i]]) 
				{
					rightChild->LinkedList[i][iterCount] = node->LinkedList[i][j];
					iterCount++;
				}
			}
			rightChild->LinkedListSize[i] = iterCount;
		}
		if (flag) {
			rightChild->quasiPiles = node->quasiPiles;
		} else {
			rightChild->split = split;
			int *newList = rightChild->LinkedList[split];
			int newListSize = rightChild->LinkedListSize[split];
			for (i=split; i<dimensionMinusOne - 1; i++) {
				rightChild->LinkedList[i] = rightChild->LinkedList[i + 1];
				rightChild->LinkedListSize[i] = rightChild->LinkedListSize[i + 1];
			}
			rightChild->LinkedList[dimensionMinusOne - 1] = newList;
			rightChild->LinkedListSize[dimensionMinusOne - 1] = newListSize;
		}
	} else {
		delete [] rightChild->partialCoverIndex;
		delete rightChild;
		rightChild = nullptr;
	}
	for (i=0; i<dimensionMinusOne; i++) {
		delete [] node->LinkedList[i];
	}
	delete [] node->LinkedListSize;
	delete [] node->partialCoverIndex;

	// recursive call for children
	buildTree(leftChild);
	if (rightChild != nullptr) {
		buildTree(rightChild);
	}
	node->leftChild = leftChild;
	node->rightChild = rightChild;	
	node->indexOfMyFirstCoverBox = 0;
	node->indexOfMySecondCoverBox = 1;
}

// fully covering case is simpler than partially covering case becuase the box must fully cover all children nodes
// so there are fewer branches in this function than removePoint function

inline void removeFullyCoverBox(BFTreeNode_largeK *node, const int fatherFirstCoverBox, const int fatherSecondCoverBox) {

	if (node->secondCoverBox < removedPoint) {
		return;
	}
	// update node->indexOfMyFirstCoverBox/indexOfMySecondCoverBox
	if (node->indexOfMyFirstCoverBox < node->fullyCount) {
		if (deleted[node->fullyCoverBoxes[node->indexOfMyFirstCoverBox]]) {
			if (node->indexOfMySecondCoverBox < node->fullyCount) {
				// find the new node->indexOfMyFirstCoverBox
				while (node->indexOfMySecondCoverBox < node->fullyCount &&
					deleted[node->fullyCoverBoxes[node->indexOfMySecondCoverBox]])
				{
					node->indexOfMySecondCoverBox++;
				}
				node->indexOfMyFirstCoverBox = node->indexOfMySecondCoverBox;
				// find the new node->indexOfMySecondCoverBox
				if (node->indexOfMySecondCoverBox < node->fullyCount) {
					node->indexOfMySecondCoverBox++;
					while (node->indexOfMySecondCoverBox < node->fullyCount &&
						deleted[node->fullyCoverBoxes[node->indexOfMySecondCoverBox]])
					{
						node->indexOfMySecondCoverBox++;
					}
				}
			} else {
				node->indexOfMyFirstCoverBox = node->fullyCount;
			}
		} else {
			// node->indexOfMyFirstCoverBox does not change
			// find the new node->indexOfMySecondCoverBox
			while (node->indexOfMySecondCoverBox < node->fullyCount &&
				deleted[node->fullyCoverBoxes[node->indexOfMySecondCoverBox]])
			{
				node->indexOfMySecondCoverBox++;
			}
		}
	}
	// update node->firstCoverBox/secondCoverBox
	// note that fatherFirstCoverBox/fatherSecondCoverBox >= new node->firstCoverBox/secondCoverBox
	// complexity is of O(1) amortized
	int oldFirstCoverBox = node->firstCoverBox, oldSecondCoverBox = node->secondCoverBox;
	if (removedPoint == node->firstCoverBox || removedPoint == node->secondCoverBox) {
		if (removedPoint == node->firstCoverBox) {
			node->firstCoverBox = node->secondCoverBox;
		}
		// new firstCoverBox is min(fatherFirstCoverBox, node->fullyCoverBoxes[node->indexOfMyFirstCoverBox])
		// note that these two values cannot be the same
		// in the first case, new secondCoverBox is the latter one
		// in the second case, new secondCoverBox is min(fatherFirstCoverBox, node->fullyCoverBoxes[node->indexOfMySecondCoverBox])
		if (node->indexOfMyFirstCoverBox < node->fullyCount) {
			if (node->fullyCoverBoxes[node->indexOfMyFirstCoverBox] == node->firstCoverBox) {
				if (node->indexOfMySecondCoverBox < node->fullyCount) {
					if (node->fullyCoverBoxes[node->indexOfMySecondCoverBox] < fatherFirstCoverBox) {
						node->secondCoverBox = node->fullyCoverBoxes[node->indexOfMySecondCoverBox];
					} else {
						node->secondCoverBox = fatherFirstCoverBox;
					}
				} else {
					node->secondCoverBox = fatherFirstCoverBox;
				}
			} else {
				if (node->fullyCoverBoxes[node->indexOfMyFirstCoverBox] < fatherSecondCoverBox) {
					node->secondCoverBox = node->fullyCoverBoxes[node->indexOfMyFirstCoverBox];
				} else {
					node->secondCoverBox = fatherSecondCoverBox;
				}
			}
		} else {
			node->secondCoverBox = fatherSecondCoverBox;
		}
	}

	// removedPoint <= node->secondCoverBox, so removing it must change some node
	if (node->leftChild == nullptr && node->rightChild == nullptr) {
		int i;
		// in this case, removedPoint must not fully cover father node
		if (removedPoint == oldFirstCoverBox) {
			// use backup data to recover the state before first covering box (i.e., removedPoint) was inserted
			// we need to re-insert boxes for this node after removedPoint
			node->firstCoverContribution = 0.;
			node->insertCount = node->backupInsertCount;
			node->xLd = population[oldFirstCoverBox][dimensionMinusOne];
			for (i=0; i<node->usefulDims; i++) {
				node->A0[i] = node->backupA0[i];
				node->A1[i] = node->backupA1[i];
			}
			node->isFirstCoverBoxInserted = false;
			for (i=0; i<node->usefulDims; i++) {
				if (node->A0[i] != popsize) {
					// compute M_U^{R,x} in the paper, it may be 0
					// when lambda=1, U is the point corresponding to A1[i]
					minusOneContributions[i] = node->uselessDimContribution;
					for (int k=0; k<node->usefulDims; k++) {
						// only for the i-th dimension, k_i in the paper is 1, otherwise is 0
						if (k == i) {
							if(node->A1[i] == popsize) {
								minusOneContributions[i] *= population[node->partialCoverIndex[node->A0[i]]][node->dims[i]] - node->lowerbound[node->dims[i]];
							} else {
								minusOneContributions[i] *= population[node->partialCoverIndex[node->A0[i]]][node->dims[i]] - population[node->partialCoverIndex[node->A1[i]]][node->dims[i]];
							}
						} else {
							if (node->A0[k] == popsize) {
								minusOneContributions[i] *= node->delta[k];
							} else {
								minusOneContributions[i] *= node->upperbound[node->dims[k]] - population[node->partialCoverIndex[node->A0[k]]][node->dims[k]];
							}
						}
					}
				}
			}
			computeLeafNode(node, node->insertCount);
		} else {
			// continue updates after the old second covering box
			minusOneContributions[0] = node->uselessDimContribution;
			for (i=0; i<node->usefulDims; i++) {
				if (node->A0[i] != popsize) {
					minusOneContributions[0] *= node->upperbound[node->dims[i]] - population[node->partialCoverIndex[node->A0[i]]][node->dims[i]];
				} else {
					minusOneContributions[0] *= node->delta[i];
				}
			}
			computeLeafNode(node, node->insertCount);
		} 
		return;
	}

	removeFullyCoverBox(node->leftChild, node->firstCoverBox, node->secondCoverBox);
	if (node->rightChild != nullptr) {
		removeFullyCoverBox(node->rightChild, node->firstCoverBox, node->secondCoverBox);
	} 
}

inline void removeFullyCoverBox(BFTreeNode_smallK *node, const int fatherFirstCoverBox, const int fatherSecondCoverBox) {

	if (node->secondCoverBox < removedPoint) {
		return;
	}
	// update node->indexOfMyFirstCoverBox/indexOfMySecondCoverBox
	if (node->indexOfMyFirstCoverBox < node->fullyCount) {
		if (deleted[node->fullyCoverBoxes[node->indexOfMyFirstCoverBox]]) {
			if (node->indexOfMySecondCoverBox < node->fullyCount) {
				// find the new node->indexOfMyFirstCoverBox
				while (node->indexOfMySecondCoverBox < node->fullyCount &&
					deleted[node->fullyCoverBoxes[node->indexOfMySecondCoverBox]])
				{
					node->indexOfMySecondCoverBox++;
				}
				node->indexOfMyFirstCoverBox = node->indexOfMySecondCoverBox;
				// find the new node->indexOfMySecondCoverBox
				if (node->indexOfMySecondCoverBox < node->fullyCount) {
					node->indexOfMySecondCoverBox++;
					while (node->indexOfMySecondCoverBox < node->fullyCount &&
						deleted[node->fullyCoverBoxes[node->indexOfMySecondCoverBox]])
					{
						node->indexOfMySecondCoverBox++;
					}
				}
			} else {
				node->indexOfMyFirstCoverBox = node->fullyCount;
			}
		} else {
			// node->indexOfMyFirstCoverBox does not change
			// find the new node->indexOfMySecondCoverBox
			while (node->indexOfMySecondCoverBox < node->fullyCount &&
				deleted[node->fullyCoverBoxes[node->indexOfMySecondCoverBox]])
			{
				node->indexOfMySecondCoverBox++;
			}
		}
	}
	// update node->firstCoverBox/secondCoverBox
	// note that fatherFirstCoverBox/fatherSecondCoverBox >= new node->firstCoverBox/secondCoverBox
	// complexity is of O(1) amortized
	int oldFirstCoverBox = node->firstCoverBox, oldSecondCoverBox = node->secondCoverBox;
	if (removedPoint == node->firstCoverBox || removedPoint == node->secondCoverBox) {
		if (removedPoint == node->firstCoverBox) {
			node->firstCoverBox = node->secondCoverBox;
		}
		// new firstCoverBox is min(fatherFirstCoverBox, node->fullyCoverBoxes[node->indexOfMyFirstCoverBox])
		// note that these two values cannot be the same
		// in the first case, new secondCoverBox is the latter one
		// in the second case, new secondCoverBox is min(fatherFirstCoverBox, node->fullyCoverBoxes[node->indexOfMySecondCoverBox])
		if (node->indexOfMyFirstCoverBox < node->fullyCount) {
			if (node->fullyCoverBoxes[node->indexOfMyFirstCoverBox] == node->firstCoverBox) {
				if (node->indexOfMySecondCoverBox < node->fullyCount) {
					if (node->fullyCoverBoxes[node->indexOfMySecondCoverBox] < fatherFirstCoverBox) {
						node->secondCoverBox = node->fullyCoverBoxes[node->indexOfMySecondCoverBox];
					} else {
						node->secondCoverBox = fatherFirstCoverBox;
					}
				} else {
					node->secondCoverBox = fatherFirstCoverBox;
				}
			} else {
				if (node->fullyCoverBoxes[node->indexOfMyFirstCoverBox] < fatherSecondCoverBox) {
					node->secondCoverBox = node->fullyCoverBoxes[node->indexOfMyFirstCoverBox];
				} else {
					node->secondCoverBox = fatherSecondCoverBox;
				}
			}
		} else {
			node->secondCoverBox = fatherSecondCoverBox;
		}
	}

	// removedPoint <= node->secondCoverBox, so removing it must change some node
	if (node->leftChild == nullptr && node->rightChild == nullptr) {
		int i;
		// in this case, removedPoint must not fully cover father node
		if (removedPoint == oldFirstCoverBox) {
			// use backup data to recover the state before first covering box (i.e., removedPoint) was inserted
			// we need to re-insert boxes for this node after removedPoint
			node->firstCoverContribution = 0.;
			node->insertCount = node->backupInsertCount;
			node->xLd = population[oldFirstCoverBox][dimensionMinusOne];
			for (i=0; i<node->usefulDims; i++) {
				node->A0[i] = node->backupA0[i];
				node->A1[i] = node->backupA1[i];
			}
			node->isFirstCoverBoxInserted = false;
			for (i=0; i<node->usefulDims; i++) {
				if (node->A0[i] != popsize) {
					// compute M_U^{R,x} in the paper, it may be 0
					// when lambda=1, U is the point corresponding to A1[i]
					minusOneContributions[i] = node->uselessDimContribution;
					for (int k=0; k<node->usefulDims; k++) {
						// only for the i-th dimension, k_i in the paper is 1, otherwise is 0
						if (k == i) {
							if(node->A1[i] == popsize) {
								minusOneContributions[i] *= population[node->partialCoverIndex[node->A0[i]]][node->dims[i]] - node->lowerbound[node->dims[i]];
							} else {
								minusOneContributions[i] *= population[node->partialCoverIndex[node->A0[i]]][node->dims[i]] - population[node->partialCoverIndex[node->A1[i]]][node->dims[i]];
							}
						} else {
							if (node->A0[k] == popsize) {
								minusOneContributions[i] *= node->delta[k];
							} else {
								minusOneContributions[i] *= node->upperbound[node->dims[k]] - population[node->partialCoverIndex[node->A0[k]]][node->dims[k]];
							}
						}
					}
				}
			}
			computeLeafNode(node, node->insertCount);
		} else {
			// continue updates after the old second covering box
			minusOneContributions[0] = node->uselessDimContribution;
			for (i=0; i<node->usefulDims; i++) {
				if (node->A0[i] != popsize) {
					minusOneContributions[0] *= node->upperbound[node->dims[i]] - population[node->partialCoverIndex[node->A0[i]]][node->dims[i]];
				} else {
					minusOneContributions[0] *= node->delta[i];
				}
			}
			computeLeafNode(node, node->insertCount);
		} 
		return;
	}

	removeFullyCoverBox(node->leftChild, node->firstCoverBox, node->secondCoverBox);
	if (node->rightChild != nullptr) {
		removeFullyCoverBox(node->rightChild, node->firstCoverBox, node->secondCoverBox);
	} 
}

// If removedPoint partially covers a leaf node, we have to totally re-compute the leaf node.
// because each point partially cover at most O(n^{(d-2)/2}logn) leaf nodes,
// re-computing these nodes takes at most O(n^{(d-1}/2)}logn) time in total.

// If removedPoint fully covers a node $v$, we need to update children if and only if
// removedPoint is the first/second fully covering box of $v$. The reason is as follows:
// Consider the rightmost leaf node $s$ among all leaf nodes that are children of $v$,
// Obviously, $s$ is fully covered by a box if and only if $v$ is fully covered by the box.
// Thus, if removedPoint is the first/second fully covering box of $v$,
// it must be the first/second fully covering box of $s$ as well, and vice versa.
inline void removePoint(BFTreeNode_largeK *node, const int fatherFirstCoverBox, const int fatherSecondCoverBox) {

	if (node->secondCoverBox < removedPoint ||
		!partCovers(population[removedPoint], node->lowerbound) )
	{
		return;
	}

	// same codes as those in removeFullyCoverBox function
	if (node->indexOfMyFirstCoverBox < node->fullyCount) {
		if (deleted[node->fullyCoverBoxes[node->indexOfMyFirstCoverBox]]) {
			if (node->indexOfMySecondCoverBox < node->fullyCount) {
				while (node->indexOfMySecondCoverBox < node->fullyCount &&
					deleted[node->fullyCoverBoxes[node->indexOfMySecondCoverBox]])
				{
					node->indexOfMySecondCoverBox++;
				}
				node->indexOfMyFirstCoverBox = node->indexOfMySecondCoverBox;
				if (node->indexOfMySecondCoverBox < node->fullyCount) {
					node->indexOfMySecondCoverBox++;
					while (node->indexOfMySecondCoverBox < node->fullyCount &&
						deleted[node->fullyCoverBoxes[node->indexOfMySecondCoverBox]])
					{
						node->indexOfMySecondCoverBox++;
					}
				}
			} else {
				node->indexOfMyFirstCoverBox = node->fullyCount;
			}
		} else {
			while (node->indexOfMySecondCoverBox < node->fullyCount &&
				deleted[node->fullyCoverBoxes[node->indexOfMySecondCoverBox]])
			{
				node->indexOfMySecondCoverBox++;
			}
		}
	}
	int oldFirstCoverBox = node->firstCoverBox, oldSecondCoverBox = node->secondCoverBox;
	if (removedPoint == node->firstCoverBox || removedPoint == node->secondCoverBox) {		
		if (removedPoint == node->firstCoverBox) {
			node->firstCoverBox = node->secondCoverBox;
		}
		if (node->indexOfMyFirstCoverBox < node->fullyCount) {
			if (node->fullyCoverBoxes[node->indexOfMyFirstCoverBox] == node->firstCoverBox) {
				if (node->indexOfMySecondCoverBox < node->fullyCount) {
					if (node->fullyCoverBoxes[node->indexOfMySecondCoverBox] < fatherFirstCoverBox) {
						node->secondCoverBox = node->fullyCoverBoxes[node->indexOfMySecondCoverBox];
					} else {
						node->secondCoverBox = fatherFirstCoverBox;
					}
				} else {
					node->secondCoverBox = fatherFirstCoverBox;
				}
			} else {
				if (node->fullyCoverBoxes[node->indexOfMyFirstCoverBox] < fatherSecondCoverBox) {
					node->secondCoverBox = node->fullyCoverBoxes[node->indexOfMyFirstCoverBox];
				} else {
					node->secondCoverBox = fatherSecondCoverBox;
				}
			}
		} else {
			node->secondCoverBox = fatherSecondCoverBox;
		}
	}

	if (node->leftChild == nullptr && node->rightChild == nullptr) {
		int i;
		if (covers(population[removedPoint], node->upperbound)) {
			// same codes for the fully covering case as those in removeFullyCoverBox function
			if (removedPoint == oldFirstCoverBox) {
				node->firstCoverContribution = 0.;
				node->insertCount = node->backupInsertCount;
				node->xLd = population[oldFirstCoverBox][dimensionMinusOne];
				for (i=0; i<node->usefulDims; i++) {
					node->A0[i] = node->backupA0[i];
					node->A1[i] = node->backupA1[i];
				}
				node->isFirstCoverBoxInserted = false;
				for (i=0; i<node->usefulDims; i++) {
					if (node->A0[i] != popsize) {
						// compute M_U^{R,x} in the paper, it may be 0
						// when lambda=1, U is the point corresponding to A1[i]
						minusOneContributions[i] = node->uselessDimContribution;
						for (int k=0; k<node->usefulDims; k++) {
							// only for the i-th dimension, k_i in the paper is 1, otherwise is 0
							if (k == i) {
								if(node->A1[i] == popsize) {
									minusOneContributions[i] *= population[node->partialCoverIndex[node->A0[i]]][node->dims[i]] - node->lowerbound[node->dims[i]];
								} else {
									minusOneContributions[i] *= population[node->partialCoverIndex[node->A0[i]]][node->dims[i]] - population[node->partialCoverIndex[node->A1[i]]][node->dims[i]];
								}
							} else {
								if (node->A0[k] == popsize) {
									minusOneContributions[i] *= node->delta[k];
								} else {
									minusOneContributions[i] *= node->upperbound[node->dims[k]] - population[node->partialCoverIndex[node->A0[k]]][node->dims[k]];
								}
							}
						}
					}
				}
				computeLeafNode(node, node->insertCount);
			} else {
				minusOneContributions[0] = node->uselessDimContribution;
				for (i=0; i<node->usefulDims; i++) {
					if (node->A0[i] != popsize) {
						minusOneContributions[0] *= node->upperbound[node->dims[i]] - population[node->partialCoverIndex[node->A0[i]]][node->dims[i]];
					} else {
						minusOneContributions[0] *= node->delta[i];
					}
				}
				computeLeafNode(node, node->insertCount);
			}
		} else {
			// re-compute the leaf node
			if (node->firstCoverBox < removedPoint) {
				// if removedPoint is inserted after the first covering box (note that the first covering box is not removed in this case)
				// we do not need to re-compute the leaf node
				// we can call computeLeafNode() after the first covering box is inserted
				node->insertCount = node->backupInsertCount;
				node->xLd = population[node->firstCoverBox][dimensionMinusOne];
				for (i=0; i<node->usefulDims; i++) {
					node->A0[i] = node->backupA0[i];
				}
				contributions[node->firstCoverBox] -= node->firstCoverContribution;
				node->firstCoverContribution = 0.;
				minusOneContributions[0] = node->uselessDimContribution;
				for (i=0; i<node->usefulDims; i++) {
					if (node->A0[i] != popsize) {
						minusOneContributions[0] *= node->upperbound[node->dims[i]] - population[node->partialCoverIndex[node->A0[i]]][node->dims[i]];
					} else {
						minusOneContributions[0] *= node->delta[i];
					}
				}
				computeLeafNode(node, node->insertCount);
			} else {
				if (oldFirstCoverBox != popsize) {
					contributions[oldFirstCoverBox] -= node->firstCoverContribution;
					node->firstCoverContribution = 0.;
					node->isFirstCoverBoxInserted = false;
					for (i=0; i<node->backupInsertCount; i++) {
						contributions[node->partialCoverIndex[i]] -= node->nodeContributions[i];
						node->nodeContributions[i] = 0.;
					}
				} else {		
					for (i=0; i<node->insertCount; i++) {
						contributions[node->partialCoverIndex[i]] -= node->nodeContributions[i];
						node->nodeContributions[i] = 0.;
					}			
				}
				for (i=0; i<node->usefulDims; i++) {
					node->A0[i] = popsize;
					node->A1[i] = popsize;
				}
				computeLeafNode(node, 0);
			}
		}
		return;
	}
	if (covers(population[removedPoint], node->upperbound)) {
		removeFullyCoverBox(node->leftChild, node->firstCoverBox, node->secondCoverBox);
		if (node->rightChild != nullptr) {
			removeFullyCoverBox(node->rightChild, node->firstCoverBox, node->secondCoverBox);
		}
	} else {
		removePoint(node->leftChild, node->firstCoverBox, node->secondCoverBox);
		if (node->rightChild != nullptr) {
			removePoint(node->rightChild, node->firstCoverBox, node->secondCoverBox);
		}
	}
}

inline void removePoint(BFTreeNode_smallK *node, const int fatherFirstCoverBox, const int fatherSecondCoverBox) {

	if (node->secondCoverBox < removedPoint ||
		!partCovers(population[removedPoint], node->lowerbound) )
	{
		return;
	}

	// same codes as those in removeFullyCoverBox function
	if (node->indexOfMyFirstCoverBox < node->fullyCount) {
		if (deleted[node->fullyCoverBoxes[node->indexOfMyFirstCoverBox]]) {
			if (node->indexOfMySecondCoverBox < node->fullyCount) {
				while (node->indexOfMySecondCoverBox < node->fullyCount &&
					deleted[node->fullyCoverBoxes[node->indexOfMySecondCoverBox]])
				{
					node->indexOfMySecondCoverBox++;
				}
				node->indexOfMyFirstCoverBox = node->indexOfMySecondCoverBox;
				if (node->indexOfMySecondCoverBox < node->fullyCount) {
					node->indexOfMySecondCoverBox++;
					while (node->indexOfMySecondCoverBox < node->fullyCount &&
						deleted[node->fullyCoverBoxes[node->indexOfMySecondCoverBox]])
					{
						node->indexOfMySecondCoverBox++;
					}
				}
			} else {
				node->indexOfMyFirstCoverBox = node->fullyCount;
			}
		} else {
			while (node->indexOfMySecondCoverBox < node->fullyCount &&
				deleted[node->fullyCoverBoxes[node->indexOfMySecondCoverBox]])
			{
				node->indexOfMySecondCoverBox++;
			}
		}
	}
	int oldFirstCoverBox = node->firstCoverBox, oldSecondCoverBox = node->secondCoverBox;
	if (removedPoint == node->firstCoverBox || removedPoint == node->secondCoverBox) {		
		if (removedPoint == node->firstCoverBox) {
			node->firstCoverBox = node->secondCoverBox;
		}
		if (node->indexOfMyFirstCoverBox < node->fullyCount) {
			if (node->fullyCoverBoxes[node->indexOfMyFirstCoverBox] == node->firstCoverBox) {
				if (node->indexOfMySecondCoverBox < node->fullyCount) {
					if (node->fullyCoverBoxes[node->indexOfMySecondCoverBox] < fatherFirstCoverBox) {
						node->secondCoverBox = node->fullyCoverBoxes[node->indexOfMySecondCoverBox];
					} else {
						node->secondCoverBox = fatherFirstCoverBox;
					}
				} else {
					node->secondCoverBox = fatherFirstCoverBox;
				}
			} else {
				if (node->fullyCoverBoxes[node->indexOfMyFirstCoverBox] < fatherSecondCoverBox) {
					node->secondCoverBox = node->fullyCoverBoxes[node->indexOfMyFirstCoverBox];
				} else {
					node->secondCoverBox = fatherSecondCoverBox;
				}
			}
		} else {
			node->secondCoverBox = fatherSecondCoverBox;
		}
	}

	if (node->leftChild == nullptr && node->rightChild == nullptr) {
		int i;
		if (covers(population[removedPoint], node->upperbound)) {
			// same codes for the fully covering case as those in removeFullyCoverBox function
			if (removedPoint == oldFirstCoverBox) {
				node->firstCoverContribution = 0.;
				node->insertCount = node->backupInsertCount;
				node->xLd = population[oldFirstCoverBox][dimensionMinusOne];
				for (i=0; i<node->usefulDims; i++) {
					node->A0[i] = node->backupA0[i];
					node->A1[i] = node->backupA1[i];
				}
				node->isFirstCoverBoxInserted = false;
				for (i=0; i<node->usefulDims; i++) {
					if (node->A0[i] != popsize) {
						// compute M_U^{R,x} in the paper, it may be 0
						// when lambda=1, U is the point corresponding to A1[i]
						minusOneContributions[i] = node->uselessDimContribution;
						for (int k=0; k<node->usefulDims; k++) {
							// only for the i-th dimension, k_i in the paper is 1, otherwise is 0
							if (k == i) {
								if(node->A1[i] == popsize) {
									minusOneContributions[i] *= population[node->partialCoverIndex[node->A0[i]]][node->dims[i]] - node->lowerbound[node->dims[i]];
								} else {
									minusOneContributions[i] *= population[node->partialCoverIndex[node->A0[i]]][node->dims[i]] - population[node->partialCoverIndex[node->A1[i]]][node->dims[i]];
								}
							} else {
								if (node->A0[k] == popsize) {
									minusOneContributions[i] *= node->delta[k];
								} else {
									minusOneContributions[i] *= node->upperbound[node->dims[k]] - population[node->partialCoverIndex[node->A0[k]]][node->dims[k]];
								}
							}
						}
					}
				}
				computeLeafNode(node, node->insertCount);
			} else {
				minusOneContributions[0] = node->uselessDimContribution;
				for (i=0; i<node->usefulDims; i++) {
					if (node->A0[i] != popsize) {
						minusOneContributions[0] *= node->upperbound[node->dims[i]] - population[node->partialCoverIndex[node->A0[i]]][node->dims[i]];
					} else {
						minusOneContributions[0] *= node->delta[i];
					}
				}
				computeLeafNode(node, node->insertCount);
			}
		} else {
			// re-compute the leaf node
			if (node->firstCoverBox < removedPoint) {
				// if removedPoint is inserted after the first covering box (note that the first covering box is not removed in this case)
				// we do not need to re-compute the leaf node
				// we can call computeLeafNode() after the first covering box is inserted
				node->insertCount = node->backupInsertCount;
				node->xLd = population[node->firstCoverBox][dimensionMinusOne];
				for (i=0; i<node->usefulDims; i++) {
					node->A0[i] = node->backupA0[i];
				}
				contributions[node->firstCoverBox] -= node->firstCoverContribution;
				node->firstCoverContribution = 0.;
				minusOneContributions[0] = node->uselessDimContribution;
				for (i=0; i<node->usefulDims; i++) {
					if (node->A0[i] != popsize) {
						minusOneContributions[0] *= node->upperbound[node->dims[i]] - population[node->partialCoverIndex[node->A0[i]]][node->dims[i]];
					} else {
						minusOneContributions[0] *= node->delta[i];
					}
				}
				computeLeafNode(node, node->insertCount);
			} else {
				if (oldFirstCoverBox != popsize) {
					contributions[oldFirstCoverBox] -= node->firstCoverContribution;
					node->firstCoverContribution = 0.;
					node->isFirstCoverBoxInserted = false;
					for (i=0; i<node->backupInsertCount; i++) {
						contributions[node->partialCoverIndex[i]] -= node->nodeContributions[i];
						node->nodeContributions[i] = 0.;
					}
				} else {		
					for (i=0; i<node->insertCount; i++) {
						contributions[node->partialCoverIndex[i]] -= node->nodeContributions[i];
						node->nodeContributions[i] = 0.;
					}			
				}
				for (i=0; i<node->usefulDims; i++) {
					node->A0[i] = popsize;
					node->A1[i] = popsize;
				}
				computeLeafNode(node, 0);
			}
		}
		return;
	}
	if (covers(population[removedPoint], node->upperbound)) {
		removeFullyCoverBox(node->leftChild, node->firstCoverBox, node->secondCoverBox);
		if (node->rightChild != nullptr) {
			removeFullyCoverBox(node->rightChild, node->firstCoverBox, node->secondCoverBox);
		}
	} else {
		removePoint(node->leftChild, node->firstCoverBox, node->secondCoverBox);
		if (node->rightChild != nullptr) {
			removePoint(node->rightChild, node->firstCoverBox, node->secondCoverBox);
		}
	}
}

int main(int  argc, char  *argv[]) {

	int i, j, k;

	/* check parameters */
	if (argc < 6)  {
		fprintf(stderr, "usage: gHSSP <number of points> <dimension> <number of points to be reserved> <input file> <reference point file> <outputfile(optional)>\n");
		exit(1);
	}
	sscanf(argv[1], "%d", &popsize);
	sscanf(argv[2], "%d", &dimension);
	sscanf(argv[3], "%d", &reservedNumber);
	dimensionMinusOne = dimension - 1;
	char *filenameData = argv[4];
	char *filenameRef = argv[5];

	/* read in data */
	char word[30];

	// read in reference point
	static double* ref = new double[dimension];
	ifstream fileRef;
	fileRef.open(filenameRef, ios::in);
	if (!fileRef.good()) {
		cout<<filenameRef<<endl;
		printf("reference point file not found \n");
		exit(0);
	}
	for (i=0; i<dimension; i++) {
		fileRef >> word;
		ref[i] = atof(word);
	}
	fileRef.close();

	// read in data file
	ifstream fileData;
	fileData.open(filenameData, ios::in);
	if (!fileData.good()){
		printf("data file not found \n");
		exit(0);
	}
	population = vector<double*>(popsize);
	for (i=0; i<popsize; i++) {
		population[i] = new double[dimension];
		for (j=0; j<dimension; j++) {
			fileData >> word;
			population[i][j] = ref[j] - atof(word);
		}
	}
	fileData.close();

	struct timeval tv1, tv2;
	struct rusage ru_before, ru_after;

	getrusage (RUSAGE_SELF, &ru_before);

	// sqrt of popsize
	dSqrtDataNumber = sqrt((double)popsize);

	int *sorted_height = new int[popsize];
	for (i=0; i<popsize; i++) {
		sorted_height[i] = i;
	}
	Index_Descend_Sort(population, sorted_height, popsize, dimensionMinusOne);
	sort(population.begin(), population.end(), Yildiz_cmp);

	contributions = new double[popsize];
	boundaries = new int[popsize];
	noBoundaries = new int[popsize];
	piles = new int[popsize];
	dimensionPileCount = new int[dimensionMinusOne]; 
	minusOneContributions = new double[dimensionMinusOne];
	for (i = 0; i < popsize; i++) {
		contributions[i] = 0.;
	}
	alter = 0;

	vector<bool> selected = vector<bool>(popsize, false);
	if (reservedNumber >= popsize * 0.9) {
		// for large k
		// root node
		BFTreeNode_largeK *root = new BFTreeNode_largeK();
		root->split = 0;
		root->partialCoverNumber = popsize;
		root->partialCoverIndex = new int[popsize];
		root->dominatedCount = new int[popsize];
		root->lowerbound = new double[dimensionMinusOne];
		root->upperbound = new double[dimensionMinusOne];
		root->dims = vector<int>(dimensionMinusOne);
		for (i=0; i<popsize; i++) {
			root->partialCoverIndex[i] = i;
			root->dominatedCount[i] = 0;
		}
		root->fullyCoverBoxes = new int[popsize];
		root->fullyCount = 0;
		root->firstCoverBox = popsize;
		root->secondCoverBox = popsize;
		root->LinkedList = vector<int*>(dimensionMinusOne);
		root->LinkedListSize = new int[dimensionMinusOne];
		for (i=0; i<dimensionMinusOne; i++) {
			root->LinkedList[i] = new int[popsize];
			for (j=0; j<popsize; j++) {
				root->LinkedList[i][j] = j;
			}
			Index_Ascend_Sort(population, root->LinkedList[i], popsize, i);
		}
		for (i=0; i<dimensionMinusOne; i++) {
			root->lowerbound[i] = 0.;
			root->upperbound[i] = population[root->LinkedList[i][popsize-1]][i];
			root->dims[i] = i;
			int start = 0, end = popsize - 1;
			while (population[root->LinkedList[i][start]][i] <= root->lowerbound[i]) {
				start++;
			}
			while (population[root->LinkedList[i][end]][i] >= root->upperbound[i]) {
				end--;
			}
			if (start != 0) {
				for (j=start; j<=end; j++) {
					root->LinkedList[i][j - start] = root->LinkedList[i][j];
				}
			}
			root->LinkedListSize[i] = end - start + 1;
		}

		// build tree
		removedPoint = -1;
		deleted = vector<bool>(popsize, false);
		buildTree(root);
		
		// execute gHSSD
		int index = 0;
		double minContribution = contributions[0];
		for (i=1; i<popsize; i++) {
			if (contributions[i] < minContribution) {
				index = i;
				minContribution = contributions[i];
			}
		} 
		deleted[index] = true;
		for (i=1; i<popsize-reservedNumber; i++) {
			removedPoint = index;
			removePoint(root, popsize, popsize);
			for (j=0; j<popsize; j++) {
				if (!deleted[j]) {
					index = j;
					minContribution = contributions[j];
					break;
				}
			}
			for (; j<popsize; j++) {
				if (!deleted[j] && contributions[j] < minContribution) {
					index = j;
					minContribution = contributions[j];
				}
			}
			deleted[index] = true;
		}
	} else {		
		BFTreeNode_smallK *root = new BFTreeNode_smallK();
		root->isQuasiPile = false;
		root->split = 0;
		root->partialCoverNumber = popsize;
		root->partialCoverIndex = new int[popsize];
		root->lowerbound = new double[dimensionMinusOne];
		root->upperbound = new double[dimensionMinusOne];
		root->dims = vector<int>(dimensionMinusOne);
		for (i=0; i<popsize; i++) {
			root->partialCoverIndex[i] = i;
		}
		root->fullyCoverBoxes = new int[popsize];
		root->fullyCount = 0;
		root->firstCoverBox = popsize;
		root->secondCoverBox = popsize;
		root->LinkedList = vector<int*>(dimensionMinusOne);
		root->LinkedListSize = new int[dimensionMinusOne];
		for (i=0; i<dimensionMinusOne; i++) {
			root->LinkedList[i] = new int[popsize];
			for (j=0; j<popsize; j++) {
				root->LinkedList[i][j] = j;
			}
			Index_Ascend_Sort(population, root->LinkedList[i], popsize, i);
		}
		for (i=0; i<dimensionMinusOne; i++) {
			root->lowerbound[i] = 0.;
			root->upperbound[i] = population[root->LinkedList[i][popsize-1]][i];
			root->dims[i] = i;
			int start = 0, end = popsize - 1;
			while (population[root->LinkedList[i][start]][i] <= root->lowerbound[i]) {
				start++;
			}
			while (population[root->LinkedList[i][end]][i] >= root->upperbound[i]) {
				end--;
			}
			if (start != 0) {
				for (j=start; j<=end; j++) {
					root->LinkedList[i][j - start] = root->LinkedList[i][j];
				}
			}
			root->LinkedListSize[i] = end - start + 1;
		}

		// build tree
		removedPoint = -1;
		deleted = vector<bool>(popsize, false);
		buildTree(root);	

		// assume that smallest contribution is smaller than 1e30
		int index = 0;
		double minContribution = contributions[0];
		for (i=1; i<popsize; i++) {
			if (contributions[i] < minContribution) {
				index = i;
				minContribution = contributions[i];
			}
		} 

		deleted[index] = true;
		for (i = 1; i < popsize - reservedNumber; i++) {
			removedPoint = index;
			removePoint(root, popsize, popsize);
			for (j=0; j<popsize; j++) {
				if (!deleted[j]) {
					index = j;
					minContribution = contributions[j];
					break;
				}
			}
			for (; j<popsize; j++) {
				if (!deleted[j] && contributions[j] < minContribution) {
					index = j;
					minContribution = contributions[j];
				}
			}
			deleted[index] = true;
		}
	}

	// find indices of remaining points in the original order
	for (i=0; i<popsize; i++) {
		if (!deleted[i]) {
			selected[sorted_height[i]] = true;
		}
	}

	getrusage (RUSAGE_SELF, &ru_after);

	tv1 = ru_before.ru_utime;
	tv2 = ru_after.ru_utime;

	if (argc >= 7) {
		ofstream myoutput(argv[6]);
		if (myoutput.fail()) {
			printf("output data file open failed \n");
			exit(0);
		}
		for (i=0; i<popsize; i++) {
			if (selected[i]) {
				// Note: index starts from 0
				myoutput << i << endl;
			}
		}
		// the last line outputs the running time
		myoutput << setprecision(8) << tv2.tv_sec + tv2.tv_usec * 1e-6 - tv1.tv_sec - tv1.tv_usec * 1e-6;
		myoutput.close();
	} else {
		for (i=0; i<popsize; i++) {
			if (selected[i]) {
				printf("%d\n", i);
			}
		}
		printf("Time(s): %.10g\n", tv2.tv_sec + tv2.tv_usec * 1e-6 - tv1.tv_sec - tv1.tv_usec * 1e-6);
		printf("memory peak: %.1fMB\n", ru_after.ru_maxrss/(double) 1000);
	}

	return 0;
}
