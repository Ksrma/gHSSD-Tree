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

struct BFTreeNode
{
	// data for building tree
	// LinkedList will be free after building tree
	int split, partialCoverNumber;
	bool isNewLinkedList;
	int *partialCoverIndex;
	double *lowerbound, *upperbound;
	BFTreeNode *leftChild, *rightChild;
	vector<int*> LinkedList;
	vector<int> dims;								

	// first box and second box among all boxes that fully cover this node (they may fully cover father nodes)
	int firstCoverBox, secondCoverBox;

	// data to maintain first covering boxes and second covering boxes belonging to this node
	// that is, boxes that fully cover father node will only be recorded in father node instead of this node
	// (otherwise, the worst-case time/space complexity would be unknown)
	int fullyCount, *fullyCoverBoxes;
	int indexOfMyFirstCoverBox, indexOfMySecondCoverBox;

	// data for computing contributions (valid for leaf nodes)
	// "backup*" record the contribution data before the first covering box is inserted
	bool isFirstCoverBoxInserted;
	int insertCount, backupInsertCount;
	double xLd, backupXLd, firstCoverContribution;
	// A0 and A1 are the first and second column of Matrix A in the paper (since lambda=1 in this program)
	// if the i-th element of A0/A1 is equal to -1, it means a fully covering box,
	// which is recorded by firstCoverBox & firstCoverContribution;
	// if it is equal to popsize, it means no pile along such dimension;
	// otherwise, it is equal to the index of i-pile, noting that this index is
	// regarding to this node (cumulated by insertCount), not the index of regarding to the whole point set.
	int *A0, *A1, *backupA0, *backupA1;
	double *nodeContributions, *backupContributions;
};

static int dimension;
static int popsize;
static int dSqrtDataNumber;
static int removedPoint;
static int *piles;
static int *boundaries;
static int *noBoundaries;
static int *dimensionPileCount;
static bool *projection;
static double *contributions;
static BFTreeNode *root;
static vector<bool> deleted;
static vector<double*> population;

inline bool Yildiz_cmp(double* a, double* b) {
	return (a[dimension-1] > b[dimension-1]);
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
	for (i=0; i<dimension-1; i++) {
		if (cub[i] < regUp[i]) {
			return false;
		}
	}
	return true;
}

inline bool partCovers(const double* cub, double *regLow) {
	static int i;
	for (i=0; i<dimension-1; i++) {
		if (cub[i] <= regLow[i]) {
			return false;
		}
	}
	return true;
}

inline int containsBoundary(const double* cub, const double regUp[], const int split, const vector<int> &order) {
	// condition only checked for split>0
	if (regUp[order[split]] <= cub[order[split]]){
		// boundary in dimension split not contained in region, thus
		// boundary is no candidate for the splitting line
		return -1;
	} 
	else {
		static int j;
		for (j=0; j<split; j++) { // check boundaries
			if (regUp[order[j]] > cub[order[j]]) {
				// boundary contained in region
				return 1;
			}
		}
	}
	// no boundary contained in region
	return 0;
}

inline int isPile(const double* cub, double *regUp) {
	static int pile;
	static int k;

	pile = dimension;
	// check all dimensions of the node
	for (k=0; k<dimension-1; k++) {
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

// insert boxes for leaf nodes from the start-th partially covering box
inline void computeLeafNode(BFTreeNode *node, const int start) {

	int i, j, k, p, id = -1;

	double HVC, BracketA_ki, BracketA_ki1; // (A_{k_i}^i)_i and (A_{k_i+1}^i)_i in the paper
	for (j=start; j<node->partialCoverNumber; j++) {
		// skip deleted points, this will not increase the complexity 
		// because the complexity is computed assuming that all boxes will be inserted
		// TODO: consider a clever method of updating node->partialCoverIndex to avoid these checks
		while (j<node->partialCoverNumber && deleted[node->partialCoverIndex[j]]) {
			j++;
			node->insertCount++;
		}
		if (j == node->partialCoverNumber) {
			break;
		}
		id = node->partialCoverIndex[j];
		// check if a fully covering box is inserted before id
		if (!node->isFirstCoverBoxInserted && node->firstCoverBox < id) {
			j--;
			id = node->firstCoverBox;
			node->isFirstCoverBoxInserted = true;
			// backup the state before the first covering box is inserted.
			// only contribution of the dominating pile in each dimension was affected by the first fully covering box,
			// so backupContributions only needs to record (dimension-1) contributions.
			// after the first covering box is inserted, all the piles have zero contribution,
			// so we do not need to backup contributions of boxes inserted after the fully covering box.
			node->backupInsertCount = node->insertCount;
			node->backupXLd = node->xLd;
			for (i=0; i<dimension-1; i++) {
				node->backupA0[i] = node->A0[i];
				node->backupA1[i] = node->A1[i];
				if (node->A0[i] != popsize) {
					node->backupContributions[node->A0[i]] = node->nodeContributions[node->A0[i]];
				}
			}
		} else if (node->secondCoverBox < id) {
			// we do not update any data for second fully covering box,
			// because no box in this node has nonzero contribution after it is inserted.
			// thus, we can stop the update for this node (i.e., break the for loop).
			// in this way, if we have to remove this second covering box in the greedy HSSP,
			// we can directly continue the updates for boxes after second fully covering box
			id = node->secondCoverBox;
			break;
		}
		// For each box x, traverse all possible positive integer vectors (k_1, k_2, ..., k_{d-1})
		// such that sum{k_i}<=lambda. Here we use lambda=1, so only one of k_i can be 1
		// so we use a loop from 0 to d-2 respectively for (1, 0, ..., 0),
		// (0, 1, ... , 0), ..., (0, ..., 0, 1) -- they are all possible vectors (k_1, k_2, ..., k_{d-1}).
		if (node->A0[0] != popsize) {
			// compute HCx in the paper when k=0
			if (node->A0[0] == -1) {
				BracketA_ki = node->upperbound[0];
				if(node->A1[0] == popsize) {
					BracketA_ki1 = node->lowerbound[0];
				} else {
					BracketA_ki1 = population[node->partialCoverIndex[node->A1[0]]][0];
				}
			} else {
				BracketA_ki = population[node->partialCoverIndex[node->A0[0]]][0];
				if(node->A1[0] == popsize) {
					BracketA_ki1 = node->lowerbound[0];
				} else {
					BracketA_ki1 = population[node->partialCoverIndex[node->A1[0]]][0];
				}
			}
			HVC = (node->xLd - population[id][dimension-1]) * (BracketA_ki - BracketA_ki1);
			// compute HCx when k=1, ..., dimension-2
			for (k=1; k<dimension-1; k++) {
				// since k <> 0, k_i in the paper is 1, otherwise is 0
				BracketA_ki = node->upperbound[k];
				if (node->A0[k] == popsize) {
					BracketA_ki1 = node->lowerbound[k];
				} else {
					// node->A0[k] <> -1 because k <> 0
					BracketA_ki1 = population[node->partialCoverIndex[node->A0[k]]][k];
				}
				HVC *= BracketA_ki - BracketA_ki1;
			}
			if (node->A0[0] == -1) {
				contributions[node->firstCoverBox] += HVC;
				node->firstCoverContribution += HVC;
			} else {
				contributions[node->partialCoverIndex[node->A0[0]]] += HVC;
				node->nodeContributions[node->A0[0]] += HVC;
			}
		}
		// cumulate contributions for the non-dominated boxes (if exist) of the 2nd, 3rd, ..., (dimension-1)-th dimensions
		// they cannot be fully covering boxes
		for (i=1; i<dimension-1; i++) {
			if (node->A0[i] != popsize) {
				// compute M_U^{R,x} in the paper, it may be 0
				// when lambda=1, U is the point corresponding to A1[i]
				HVC = node->xLd - population[id][dimension-1];
				for (k=0; k<dimension-1; k++) {
					// only for the i-th dimension, k_i in the paper is 1, otherwise is 0
					if (k == i) {
						// node->A0[i] <> -1 since i <> 0
						BracketA_ki = population[node->partialCoverIndex[node->A0[i]]][i];
						if(node->A1[i] == popsize) {
							BracketA_ki1 = node->lowerbound[i];
						} else {
							BracketA_ki1 = population[node->partialCoverIndex[node->A1[i]]][i];
						}
					} else {
						BracketA_ki = node->upperbound[k];
						if (node->A0[k] == popsize) {
							BracketA_ki1 = node->lowerbound[k];
						} else {
							if (node->A0[k] == -1) {
								BracketA_ki1 = node->upperbound[k];
							} else {
								BracketA_ki1 = population[node->partialCoverIndex[node->A0[k]]][k];
							}
						}
					}
					HVC *= BracketA_ki - BracketA_ki1;
				}
				contributions[node->partialCoverIndex[node->A0[i]]] += HVC;
				node->nodeContributions[node->A0[i]] += HVC;
			}
		}
		if (id == node->firstCoverBox) {
			// for the first covering box, update arbitrary one dimension. 
			// in this program, we select the first dimension.
			node->A1[0] = node->A0[0];
			node->A0[0] = -1;
		} else {
			// update A[p] by population[id][p] if it is a p-pile.
			p = isPile(population[id], node->upperbound);
			if (node->A0[p] == popsize) {
				// actually we can use j here and update node->insertCount after the loop
				// but this is just a tiny improvement
				node->A0[p] = node->insertCount;
			} else {
				if (node->A0[p] == -1) {
					if (node->A1[p] == popsize) {
						node->A1[p] = node->insertCount;
					} else {
						if (population[id][p] > population[node->partialCoverIndex[node->A1[p]]][p]) {
							node->A1[p] = node->insertCount;
						}
					}
				} else {
					if (population[id][p] > population[node->partialCoverIndex[node->A0[p]]][p]) {
						node->A1[p] = node->A0[p];
						node->A0[p] = node->insertCount;
					} else {
						if (node->A1[p] == popsize) {
							node->A1[p] = node->insertCount;
						} else {
							if (population[id][p] > population[node->partialCoverIndex[node->A1[p]]][p]) {
								node->A1[p] = node->insertCount;
							}
						}
					}
				}
			}
			node->insertCount++;
		}
		node->xLd = population[id][dimension-1];
	}
	// the first covering box may be inserted after all the partially covering box
	if (!node->isFirstCoverBoxInserted && node->firstCoverBox != popsize) {
		id = node->firstCoverBox;
		node->isFirstCoverBoxInserted = true;
		node->backupInsertCount = node->insertCount;
		node->backupXLd = node->xLd;
		for (i=0; i<dimension-1; i++) {
			node->backupA0[i] = node->A0[i];
			node->backupA1[i] = node->A1[i];
			if (node->A0[i] != popsize) {
				node->backupContributions[node->A0[i]] = node->nodeContributions[node->A0[i]];
			}
		}
		for (i=0; i<dimension-1; i++) {
			if (node->A0[i] != popsize) {
				HVC = node->xLd - population[id][dimension-1];
				for (k=0; k<dimension-1; k++) {
					if (k == i) {
						BracketA_ki = population[node->partialCoverIndex[node->A0[i]]][i];
						if(node->A1[i] == popsize) {
							BracketA_ki1 = node->lowerbound[i];
						} else {
							BracketA_ki1 = population[node->partialCoverIndex[node->A1[i]]][i];
						}
					} else {
						BracketA_ki = node->upperbound[k];
						if (node->A0[k] == popsize) {
							BracketA_ki1 = node->lowerbound[k];
						} else {
							if (node->A0[k] == -1) {
								BracketA_ki1 = node->upperbound[k];
							} else {
								BracketA_ki1 = population[node->partialCoverIndex[node->A0[k]]][k];
							}
						}
					}
					HVC *= BracketA_ki - BracketA_ki1;
				}
				contributions[node->partialCoverIndex[node->A0[i]]] += HVC;
				node->nodeContributions[node->A0[i]] += HVC;
			}
		}
		node->A1[0] = node->A0[0];
		node->A0[0] = -1;
		node->xLd = population[id][dimension-1];
	}
	// two possible cases for second covering box
	double base;
	if (node->secondCoverBox == popsize) {
		// compute the contribution until the last box ({(0, ..., 0)} in the set S in BF's paper)
		// i.e., sweep from xLd to 0
		base = node->xLd;
	} else {
		// in this case, we must have id == node->secondCoverBox
		// i.e., the previous for loop ended due to this,
		// so we sweep from xLd to this box and do not change any other data
		base = node->xLd - population[node->secondCoverBox][dimension-1];
		node->xLd = population[node->secondCoverBox][dimension-1];
	}
	HVC = base;
	if (node->A0[0] != popsize) {
		if (node->A0[0] == -1) {
			BracketA_ki = node->upperbound[0];
			if(node->A1[0] == popsize) {
				BracketA_ki1 = node->lowerbound[0];
			} else {
				BracketA_ki1 = population[node->partialCoverIndex[node->A1[0]]][0];
			}
		} else {
			BracketA_ki = population[node->partialCoverIndex[node->A0[0]]][0];
			if(node->A1[0] == popsize) {
				BracketA_ki1 = node->lowerbound[0];
			} else {
				BracketA_ki1 = population[node->partialCoverIndex[node->A1[0]]][0];
			}
		}
		HVC *= BracketA_ki - BracketA_ki1;
		for (k=1; k<dimension-1; k++) {
			BracketA_ki = node->upperbound[k];
			if (node->A0[k] == popsize) {
				BracketA_ki1 = node->lowerbound[k];
			} else {
				BracketA_ki1 = population[node->partialCoverIndex[node->A0[k]]][k];
			}
			HVC *= BracketA_ki - BracketA_ki1;
		}
		if (node->A0[0] == -1) {
			contributions[node->firstCoverBox] += HVC;
			node->firstCoverContribution += HVC;
		} else {
			contributions[node->partialCoverIndex[node->A0[0]]] += HVC;
			node->nodeContributions[node->A0[0]] += HVC;
		}
	}
	for (i=1; i<dimension-1; i++) {
		if (node->A0[i] != popsize) {
			HVC = base;
			for (k=0; k<dimension-1; k++) {
				if (k == i) {
					BracketA_ki = population[node->partialCoverIndex[node->A0[i]]][i];
					if(node->A1[i] == popsize) {
						BracketA_ki1 = node->lowerbound[i];
					} else {
						BracketA_ki1 = population[node->partialCoverIndex[node->A1[i]]][i];
					}
				} else {
					BracketA_ki = node->upperbound[k];
					if (node->A0[k] == popsize) {
						BracketA_ki1 = node->lowerbound[k];
					} else {
						if (node->A0[k] == -1) {
							BracketA_ki1 = node->upperbound[k];
						} else {
							BracketA_ki1 = population[node->partialCoverIndex[node->A0[k]]][k];
						}
					}
				}
				HVC *= BracketA_ki - BracketA_ki1;
			}
			contributions[node->partialCoverIndex[node->A0[i]]] += HVC;
			node->nodeContributions[node->A0[i]] += HVC;
		}
	}
}

inline void buildTree(BFTreeNode *node) {

	unsigned int i, j, k, l;

	int id, iterCount, count = 0, nonPiles = 0;
	int linkedListSize = node->LinkedList.size();
	for (i=0; i<dimension-1; i++) {
		dimensionPileCount[i] = 0;
	}
	for (i=0; i<node->partialCoverNumber; i++) {
		id = node->partialCoverIndex[i];
		if (covers(population[id], node->upperbound)) {
			// save first two fully covering boxes (considering father nodes)
			if (id < node->firstCoverBox) {
				node->secondCoverBox = node->firstCoverBox;
				node->firstCoverBox = id;
			} else {
				if (id < node->secondCoverBox) {
					node->secondCoverBox = id;
				}
			}
			projection[id] = false;
			// save all fully covering box of this node
			node->fullyCoverBoxes[node->fullyCount] = id;
			node->fullyCount++;
		} else {
			projection[id] = true;
			piles[id] = isPile(population[id], node->upperbound);
			if (piles[id] == -1) {
				nonPiles++;
			} else {
				dimensionPileCount[piles[id]]++;
			}
			node->partialCoverIndex[count] = id;
			count++;
		}
	}	
	
	bool dimensionFlag = true;
	for (i=0; i<dimension-1; i++) {
		if (dimensionPileCount[i] > dSqrtDataNumber) {
			dimensionFlag = false;
			break;
		}
	}

	// compute leaf node
	if (nonPiles == 0 && dimensionFlag) {

		node->isFirstCoverBoxInserted = false;
		node->indexOfMyFirstCoverBox = 0;
		node->indexOfMySecondCoverBox = 1;
		node->partialCoverNumber = count;
		node->insertCount = 0;
		node->xLd = 0.;
		node->firstCoverContribution = 0.;
		node->A0 = new int[dimension-1];
		node->A1 = new int[dimension-1];
		node->backupA0 = new int[dimension-1];
		node->backupA1 = new int[dimension-1];
		node->nodeContributions = new double[count];
		node->backupContributions = new double[count];
		for (i=0; i<dimension-1; i++) {
			node->A0[i] = popsize;	// A^i_1 in the paper, not equal to any index in partialCoverIndex before the computation, i.e., it is undefined
			node->A1[i] = popsize;	// A^i_2 in the paper
		}
		for (i=0; i<count; i++) {
			node->nodeContributions[i] = 0.;
		}
		computeLeafNode(node, 0);

		if (node->isNewLinkedList) {
			for (i=0; i<linkedListSize; i++) {
				delete [] node->LinkedList[i];
			}
		}
		return;
	}

	// build LinkedList for left children
	vector<int*> newList;
	if (count == node->partialCoverNumber) {
		newList = node->LinkedList;
	} else {
		newList = vector<int*>(linkedListSize);
		for (i=0; i<linkedListSize; i++) {
			iterCount = 0;
			newList[i] = new int[count];
			for (j=0; j<node->partialCoverNumber; j++) {
				if (projection[node->LinkedList[i][j]]) {
					newList[i][iterCount] = node->LinkedList[i][j];
					iterCount++;
				}
			}
		}
	}
	// cout<<"TEST! linkedListSize = "<<linkedListSize<<", dim: ";
	// for (i=0; i<dimension-1; i++) {
	// 	cout<<node->dims[i]<<" ";
	// }
	// cout<<endl;

	int split = node->split;
	int middleIndex = -1;
	int boundSize = 0, noBoundSize = 0;

	do {
		for (i=0; i<count; i++) {
			int contained = containsBoundary(population[newList[split - node->split][i]], node->upperbound, split, node->dims);
			if (contained == 1) {
				boundaries[boundSize] = newList[split - node->split][i];
				boundSize++;
			} else if (contained == 0) {
				noBoundaries[noBoundSize] = newList[split - node->split][i];
				noBoundSize++;
			}
		}

		if (boundSize > 0) {
			middleIndex = boundaries[boundSize/2];
		}
		else if (noBoundSize > dSqrtDataNumber) {
			middleIndex = noBoundaries[noBoundSize/2];
		}
		else {
			split++;
			noBoundSize = 0;
		}
		
	} while (middleIndex == -1);


	BFTreeNode *leftChild = new BFTreeNode();
	BFTreeNode *rightChild = new BFTreeNode();
	leftChild->split = split;
	leftChild->partialCoverNumber = count;
	leftChild->firstCoverBox = node->firstCoverBox;
	leftChild->secondCoverBox = node->secondCoverBox;
	leftChild->lowerbound = new double[dimension-1];
	leftChild->upperbound = new double[dimension-1];
	for (i=0; i<dimension-1; i++) {
		leftChild->lowerbound[i] = node->lowerbound[i];
		leftChild->upperbound[i] = node->upperbound[i];
	}
	int trueDimension = node->dims[split];
	leftChild->upperbound[trueDimension] = population[middleIndex][trueDimension];
	//cout<<population[middleIndex][trueDimension]<<endl;
	//getchar();
	leftChild->LinkedList = vector<int*>(dimension - split - 1);
	leftChild->dims = vector<int>(dimension - 1);
	// record the first split dimensions (they are fixed)
	// linkedList are not recorded for the first split dimensions
	for (i=0; i<split; i++) {
		leftChild->dims[i] = node->dims[i];
	}
	// change the order of rest dimensions
	for (i=0; i<dimension - split - 2; i++) {
		leftChild->dims[i + split] = node->dims[i + split + 1];
		leftChild->LinkedList[i] = newList[i + split - node->split + 1];
	}
	leftChild->dims[dimension - 2] = node->dims[split];
	leftChild->LinkedList[dimension - split - 2] = newList[split - node->split];
	leftChild->partialCoverIndex = new int[count];
	leftChild->fullyCoverBoxes = new int[count];
	leftChild->fullyCount = 0;
	rightChild->partialCoverIndex = new int[count];
	iterCount = 0;
	for (i=0; i<count; i++) {
		id = node->partialCoverIndex[i];
		leftChild->partialCoverIndex[i] = id;
		if (population[id][trueDimension] > population[middleIndex][trueDimension]) {
			rightChild->partialCoverIndex[iterCount] = id;
			iterCount++;
		} else {
			projection[id] = false;
		}
	}
	if (iterCount > 0 || node->firstCoverBox != popsize) {
		rightChild->split = split;
		rightChild->dims = leftChild->dims;
		rightChild->partialCoverNumber = iterCount;
		rightChild->firstCoverBox = node->firstCoverBox;
		rightChild->secondCoverBox = node->secondCoverBox;
		rightChild->fullyCoverBoxes = new int[iterCount];
		rightChild->fullyCount = 0;
		rightChild->lowerbound = new double[dimension-1];
		rightChild->upperbound = new double[dimension-1];
		for (i=0; i<dimension-1; i++) {
			rightChild->lowerbound[i] = node->lowerbound[i];
			rightChild->upperbound[i] = node->upperbound[i];
		}
		rightChild->lowerbound[trueDimension] = population[middleIndex][trueDimension];
		rightChild->isNewLinkedList = true;
		rightChild->LinkedList = vector<int*>(dimension - split - 1);
		for (i=0; i<dimension - split - 1; i++) {
			rightChild->LinkedList[i] = new int[iterCount];
			iterCount = 0;
			for (j=0; j<count; j++) {
				if (projection[leftChild->LinkedList[i][j]]) {
					rightChild->LinkedList[i][iterCount] = leftChild->LinkedList[i][j];
					iterCount++;
				}
			}
		}
	} else {
		delete [] rightChild->partialCoverIndex;
		delete rightChild;
		rightChild = NULL;
	}
	if (count != node->partialCoverNumber) {
		leftChild->isNewLinkedList = true;
		for (i=0; i<split - node->split; i++) {
			delete [] newList[i];
		}
		if (node->isNewLinkedList) {
			for (i=0; i<linkedListSize; i++) {
				delete [] node->LinkedList[i];
			}
		}
	} else {
		leftChild->isNewLinkedList = false;
	}

	// recursive call for children
	buildTree(leftChild);
	if (node->isNewLinkedList && !leftChild->isNewLinkedList) {
		for (i=0; i<linkedListSize; i++) {
			delete [] node->LinkedList[i];
		}
	}
	if (rightChild != NULL) {
		buildTree(rightChild);
	}
	node->leftChild = leftChild;
	node->rightChild = rightChild;	
	node->indexOfMyFirstCoverBox = 0;
	node->indexOfMySecondCoverBox = 1;
	delete [] node->partialCoverIndex;
}

// fully covering case is simpler than partially covering case becuase the box must fully cover all children nodes
// so there are fewer branches in this function than removePoint function
inline void removeFullyCoverBox(BFTreeNode *node, const int fatherFirstCoverBox, const int fatherSecondCoverBox) {

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
	if (node->leftChild == NULL && node->rightChild == NULL) {
		// in this case, removedPoint must not fully cover father node
		if (removedPoint == oldFirstCoverBox) {
			// use backup data to recover the state before first covering box (i.e., removedPoint) was inserted
			// we need to re-insert boxes for this node after removedPoint
			node->firstCoverContribution = 0.;
			node->insertCount = node->backupInsertCount;
			node->xLd = node->backupXLd;
			for (int i=0; i<dimension-1; i++) {
				node->A0[i] = node->backupA0[i];
				node->A1[i] = node->backupA1[i];
				if (node->A0[i] != popsize) {
					contributions[node->partialCoverIndex[node->A0[i]]] -=
						node->nodeContributions[node->A0[i]] - node->backupContributions[node->A0[i]];
					node->nodeContributions[node->A0[i]] = node->backupContributions[node->A0[i]];
				}
			}
			node->isFirstCoverBoxInserted = false;
			computeLeafNode(node, node->insertCount);
		} else if (removedPoint == oldSecondCoverBox) {
			// continue updates after the old second covering box
			computeLeafNode(node, node->insertCount);
		} 
		return;
	}

	removeFullyCoverBox(node->leftChild, node->firstCoverBox, node->secondCoverBox);
	if (node->rightChild != NULL) {
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
// This is why the claim in the first sentence holds.
inline void removePoint(BFTreeNode *node, const int fatherFirstCoverBox, const int fatherSecondCoverBox) {

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

	if (node->leftChild == NULL && node->rightChild == NULL) {
		int i;
		if (covers(population[removedPoint], node->upperbound)) {
			// same codes for the fully covering case as those in removeFullyCoverBox function
			if (removedPoint == oldFirstCoverBox) {
				node->firstCoverContribution = 0.;
				node->insertCount = node->backupInsertCount;
				node->xLd = node->backupXLd;
				// if (node->insertCount > 0) {
				// 	node->xLd = population[node->partialCoverIndex[node->insertCount-1]][dimension-1];
				// }
				for (i=0; i<dimension-1; i++) {
					node->A0[i] = node->backupA0[i];
					node->A1[i] = node->backupA1[i];
					if (node->A0[i] != popsize) {
						contributions[node->partialCoverIndex[node->A0[i]]] -=
							node->nodeContributions[node->A0[i]] - node->backupContributions[node->A0[i]];
						node->nodeContributions[node->A0[i]] = node->backupContributions[node->A0[i]];
					}
				}
				node->isFirstCoverBoxInserted = false;
				computeLeafNode(node, node->insertCount);
			} else if (removedPoint == oldSecondCoverBox) {
				computeLeafNode(node, node->insertCount);
			}
		} else {
			// re-compute the leaf node
			for (i=0; i<node->insertCount; i++) {
				contributions[node->partialCoverIndex[i]] -= node->nodeContributions[i];
				node->nodeContributions[i] = 0.;
			}
			if (oldFirstCoverBox != popsize) {
				contributions[oldFirstCoverBox] -= node->firstCoverContribution;
				node->firstCoverContribution = 0.;
			}
			node->insertCount = 0;
			node->xLd = 0.;
			for (i=0; i<dimension-1; i++) {
				node->A0[i] = popsize;
				node->A1[i] = popsize;
			}
			node->isFirstCoverBoxInserted = false;
			computeLeafNode(node, 0);
		}
		return;
	}
	if (covers(population[removedPoint], node->upperbound)) {
		removeFullyCoverBox(node->leftChild, node->firstCoverBox, node->secondCoverBox);
		if (node->rightChild != NULL) {
			removeFullyCoverBox(node->rightChild, node->firstCoverBox, node->secondCoverBox);
		}
	} else {
		removePoint(node->leftChild, node->firstCoverBox, node->secondCoverBox);
		if (node->rightChild != NULL) {
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
	sscanf(argv[3], "%d", &k);
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
	Index_Descend_Sort(population, sorted_height, popsize, dimension-1);
	sort(population.begin(), population.end(), Yildiz_cmp);

	// root node
	BFTreeNode *root = new BFTreeNode();
	root->split = 0;
	root->partialCoverNumber = popsize;
	root->partialCoverIndex = new int[popsize];
	root->lowerbound = new double[dimension-1];
	root->upperbound = new double[dimension-1];
	root->dims = vector<int>(dimension-1);
	for (i=0; i<popsize; i++) {
		root->partialCoverIndex[i] = i;
	}
	root->fullyCoverBoxes = new int[popsize];
	root->fullyCount = 0;
	root->firstCoverBox = popsize;
	root->secondCoverBox = popsize;
	root->isNewLinkedList = true;
	root->LinkedList = vector<int*>(dimension-1);
	for (i=0; i<dimension-1; i++) {
		root->LinkedList[i] = new int[popsize];
		for (j=0; j<popsize; j++) {
			root->LinkedList[i][j] = j;
		}
		Index_Ascend_Sort(population, root->LinkedList[i], popsize, i);
	}
	for (i=0; i<dimension-1; i++) {
		root->lowerbound[i] = 0.;
		root->upperbound[i] = population[root->LinkedList[i][popsize-1]][i];
		root->dims[i] = i;
	}

	contributions = new double[popsize];
	for (i=0; i<popsize; i++) {
		contributions[i] = 0.;
	}
	boundaries = new int[popsize];
	noBoundaries = new int[popsize];
	projection = new bool[popsize];
	piles = new int[popsize];
	dimensionPileCount = new int[dimension-1];

	// run stream
	deleted = vector<bool>(popsize, false);
	removedPoint = -1;
	buildTree(root);	

	// assume that smallest contribution is smaller than 1e30
	int index = -1;
	double minContribution = 1e30;
	for (i=0; i<popsize; i++) {
		if (!deleted[i] && contributions[i] < minContribution) {
			index = i;
			minContribution = contributions[i];
		}
	}
	deleted[index] = true;
	for (i=1; i<popsize-k; i++) {
		removedPoint = index;
		removePoint(root, popsize, popsize);
		index = -1;
		minContribution = 1e30;
		for (j=0; j<popsize; j++) {
			if (!deleted[j] && contributions[j] < minContribution) {
				index = j;
				minContribution = contributions[j];
			}
		}
		deleted[index] = true;
	}
	
	// find indices of remaining points in the original order
	vector<bool> selected = vector<bool>(popsize, false);
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
