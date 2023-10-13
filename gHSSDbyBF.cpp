// gHSSD by directly using BF's algorithm
// Our Implementation of BF's algorithm for HVC (of each single point), with LinkedList used in HVC4DGS

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

Karl Bringmann, Tobias Friedrich, "An Efficient Algorithm for 
Computing Hypervolume Contributions", Evolutionary Computation, vol. 18,
no. 3, pp. 383-402, 2010.

Compilation: g++ -O3 gHSSDbyBF.cpp
Usage: gHSSDbyBF <number of points> <dimension> <number of points to be reserved>
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
	int split, partialCoverNumber;
	bool isNewLinkedList;
	int *partialCoverIndex;
	double *lowerbound, *upperbound;
	vector<int> fullyCoveringBox;
	vector<int*> LinkedList;
	vector<int> dims;			
};

static int dimension;
static int popsize;
static int dSqrtDataNumber;
static int *piles;
static int *boundaries;
static int *noBoundaries;
static int* A0;	 // First column of matrix A in Algorithm 2 in BF's paper; here we save the index of the point in A
static int* A1;	 // Second column of matrix A
static bool *projection;
static double *contributions;
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

// merge two sorted arrays (no same values) into one array in once scan, O(n)
int *mergeTwoSortedList(vector<int> &list1, int *list2, int size2) {
	int size1 = list1.size();
	int *output = new int[size1+size2];
	int count1 = 0, count2 = 0, count = 0;
	while (count1 < size1 && count2 < size2) {
		if (list1[count1] < list2[count2]) {
			output[count] = list1[count1];
			count++;
			count1++;
		} else {
			output[count] = list2[count2];
			count++;
			count2++;
		}
	}
	while (count1 < size1) {
		output[count] = list1[count1];
		count++;
		count1++;
	}
	while (count2 < size2) {
		output[count] = list2[count2];
		count++;
		count2++;
	}
	return output;
}

inline void stream(BFTreeNode *node) {

	unsigned int i, j, k, l;

	int id, iterCount, count = 0, nonPiles = 0;
	int linkedListSize = node->LinkedList.size();
	vector<int> newFullyCoveringBox = node->fullyCoveringBox;
	int newSecondCover = popsize;
	if (newFullyCoveringBox.size() == 2) {
		newSecondCover = newFullyCoveringBox[1];
	}
	for (i=0; i<node->partialCoverNumber; i++) {
		id = node->partialCoverIndex[i];
		if (id > newSecondCover) {
			break;
		}
		if (covers(population[id], node->upperbound)) {
			// save first two fully covering boxes
			switch (newFullyCoveringBox.size()) {
			case 0:
				newFullyCoveringBox.push_back(id);
				break;
			case 1:
				if (newFullyCoveringBox[0] > id) {
					newFullyCoveringBox.push_back(newFullyCoveringBox[0]);
					newFullyCoveringBox[0] = id;
				} else {
					newSecondCover = id;
					newFullyCoveringBox.push_back(id);
				}
				break;
			default:
				if (newFullyCoveringBox[0] > id) {
					newSecondCover = newFullyCoveringBox[0];
					newFullyCoveringBox[1] = newFullyCoveringBox[0];
					newFullyCoveringBox[0] = id;
				} else if (newFullyCoveringBox[1] > id) {
					newSecondCover = id;
					newFullyCoveringBox[1] = id;
				}
			}
			projection[id] = false;
		} else {
			projection[id] = true;
			piles[id] = isPile(population[id], node->upperbound);
			if (piles[id] == -1) {
				nonPiles++;
			}
			node->partialCoverIndex[count] = id;
			count++;
		}
	}

	// compute leaf node
	if (nonPiles == 0) {

		int fullyCount = newFullyCoveringBox.size();
		for (i=0; i<fullyCount; i++) {
			piles[newFullyCoveringBox[i]] = dimension;
		}
		// merge partially covering boxes and at most two fully covering boxes
		// they have been in descending order in d-th dimension
		int boxCount = fullyCount + count, *allBox;
		if (fullyCount > 0) {
			allBox = mergeTwoSortedList(newFullyCoveringBox, node->partialCoverIndex, count);
		} else {
			allBox = node->partialCoverIndex;
		}

		// initialization
		double xLd = population[allBox[0]][dimension-1], HCx;
		double BracketA_ki, BracketA_ki1; // (A_{k_i}^i)_i and (A_{k_i+1}^i)_i in the paper
		for (i=0; i<dimension-1; i++) {
			A0[i] = popsize;	// A^i_1 in the paper, not equal to any index in allBox before the computation, i.e., it is undefined
			A1[i] = popsize;	// A^i_2 in the paper
		}

		for (j=0; j<boxCount; j++) {
			id = allBox[j];
			// boxes (including fully covering boxes) have zero contribution after two fully covering boxes have been inserted
			if (fullyCount == 2 && id > newFullyCoveringBox[1]) {
				break;
			}
			// For each box x(j), traverse all possible positive integer vectors (k_1, k_2, ..., k_{d-1})
			// such that sum{k_i}<=lambda. Here we use lambda=1, so only one of k_i can be 1
			// so we use a loop from 0 to d-2 respectively for (1, 0, ..., 0), 
			// (0, 1, ... , 0), ..., (0, ..., 0, 1) -- they are all possible vectors (k_1, k_2, ..., k_{d-1}).
			for (i=0; i<dimension-1; i++) {
				// (k_1, k_2, ..., k_{d-1}) = (0, ..., 0, 1, 0, ..., 0) where k_i=1
				// if A0[i] is defined, then A_{k_i}^i for all i \in [0, d-1] are defined
				// in this case we compute contribution
				if (A0[i] != popsize) {
					// compute M_U^{R,x} in the paper, it may be 0
					// when lambda=1, U is the point corresponding to A1[i]
					HCx = 1.0;
					for (k=0; k<dimension-1; k++) {
						// only for the i-th dimension, k_i in the paper is 1, otherwise is 0
						if (k == i) {
							BracketA_ki = population[A0[i]][i];
							if(A1[i] == popsize) {
								BracketA_ki1 = node->lowerbound[i];
							} else {
								BracketA_ki1 = population[A1[i]][i];
							}
						} else {
							BracketA_ki = node->upperbound[k];
							if (A0[k] == popsize) {
								BracketA_ki1 = node->lowerbound[k];
							} else {
								BracketA_ki1 = population[A0[k]][k];
							}
						}
						HCx *= min(node->upperbound[k], BracketA_ki) - min(node->upperbound[k], BracketA_ki1);
					}
					contributions[A0[i]] += (xLd - population[id][dimension-1]) * HCx;
				}
			}
			// update A[l] by population[id][l] if it is a l-pile.
			l = piles[id];
			if (l != dimension) {				
				if (A0[l] == popsize) {
					A0[l] = id;
				} else {
					if (population[id][l] > population[A0[l]][l]) {
						A1[l] = A0[l];
						A0[l] = id;
					} else {
						if (A1[l] == popsize) {
							A1[l] = id;
						} else {
							if (population[id][l] > population[A1[l]][l]) {
								A1[l] = id;
							}
						}
					}
				}
			} else {
				// for fully covering box, use lower bound of the region to update arbitrary one dimension
				if (A0[0] == popsize) {
					A0[0] = id;
				} else {
					A1[0] = A0[0];
					A0[0] = id;
				}
			}
			xLd = population[id][dimension-1];
		}
		if (fullyCount < 2) {
			// compute the contribution of the last box (corresponds to {(0, ..., 0)} in the set S in the paper)
			for (i=0; i<dimension-1; i++) {
				if (A0[i] != popsize) {
					HCx = 1.0;
					for (k=0; k<dimension-1; k++) {
						// only for the i-th dimension, k_i in the paper is 1, otherwise is 0
						if (k == i) {
							BracketA_ki = population[A0[i]][i];
							if(A1[i] == popsize) {
								BracketA_ki1 = node->lowerbound[i];
							} else {
								BracketA_ki1 = population[A1[i]][i];
							}
						} else {
							BracketA_ki = node->upperbound[k];
							if (A0[k] == popsize) {
								BracketA_ki1 = node->lowerbound[k];
							} else {
								BracketA_ki1 = population[A0[k]][k];
							}
						}
						HCx *= min(node->upperbound[k], BracketA_ki) - min(node->upperbound[k], BracketA_ki1);
					}
					contributions[A0[i]] += xLd * HCx;
				}
			}
		}
		if (fullyCount > 0) {
			delete [] allBox;
		}
		if (node->isNewLinkedList) {
			for (i=0; i<linkedListSize; i++) {
				delete [] node->LinkedList[i];
			}
		}
		delete [] node->partialCoverIndex;
		return;
	}
	
	// boxes inserted after the second fully covering box are useless for children
	for (j=i; j<node->partialCoverNumber; j++) {
		projection[node->partialCoverIndex[j]] = false;
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
	leftChild->lowerbound = new double[dimension-1];
	leftChild->upperbound = new double[dimension-1];
	for (i=0; i<dimension-1; i++) {
		leftChild->lowerbound[i] = node->lowerbound[i];
		leftChild->upperbound[i] = node->upperbound[i];
	}
	int trueDimension = node->dims[split];
	leftChild->upperbound[trueDimension] = population[middleIndex][trueDimension];
	leftChild->fullyCoveringBox = newFullyCoveringBox;
	leftChild->LinkedList = vector<int*>(dimension - split - 1);
	leftChild->dims = vector<int>(dimension - 1);
	for (i=0; i<dimension - split - 1; i++) {
		leftChild->LinkedList[i] = newList[i + split - node->split];
	}
	for (i=0; i<split; i++) {
		leftChild->dims[i] = node->dims[i];
	}
	for (i=0; i<dimension - split - 2; i++) {
		leftChild->dims[i + split] = node->dims[i + split + 1];
		leftChild->LinkedList[i] = newList[i + split - node->split + 1];
	}
	leftChild->dims[dimension - 2] = node->dims[split];
	leftChild->LinkedList[dimension - split - 2] = newList[split - node->split];
	leftChild->partialCoverIndex = new int[count];
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
	if (iterCount > 0) {
		rightChild->split = split;
		rightChild->dims = leftChild->dims;
		rightChild->partialCoverNumber = iterCount;
		rightChild->lowerbound = new double[dimension-1];
		rightChild->upperbound = new double[dimension-1];
		for (i=0; i<dimension-1; i++) {
			rightChild->lowerbound[i] = node->lowerbound[i];
			rightChild->upperbound[i] = node->upperbound[i];
		}
		rightChild->lowerbound[trueDimension] = population[middleIndex][trueDimension];
		rightChild->fullyCoveringBox = newFullyCoveringBox;
		rightChild->isNewLinkedList = true;
		rightChild->LinkedList = vector<int*>(dimension - split - 1);
		int iterCount2;
		for (i=0; i<dimension - split - 1; i++) {
			rightChild->LinkedList[i] = new int[iterCount];
			iterCount2 = 0;
			for (j=0; j<count; j++) {
				if (projection[leftChild->LinkedList[i][j]]) {
					rightChild->LinkedList[i][iterCount2] = leftChild->LinkedList[i][j];
					iterCount2++;
				}
			}
		}
	} else {
		if (newFullyCoveringBox.size() > 0) {
			// when there are only fully covering boxes for right child, directly compute the contribution
			double HVC = 1.0;
			for (i=0; i<dimension-1; i++) {
				if (i != trueDimension) {
					HVC *= node->upperbound[i] - node->lowerbound[i];
				} else {
					HVC *= node->upperbound[trueDimension] - population[middleIndex][trueDimension];
				}
			}
			if (newFullyCoveringBox.size() == 1) {
				contributions[newFullyCoveringBox[0]] += HVC * population[newFullyCoveringBox[0]][dimension-1];
			} else {
				contributions[newFullyCoveringBox[0]] += HVC * 
					(population[newFullyCoveringBox[0]][dimension-1] - population[newFullyCoveringBox[1]][dimension-1]);
			}
		}
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
	stream(leftChild);
	if (node->isNewLinkedList && !leftChild->isNewLinkedList) {
		for (i=0; i<linkedListSize; i++) {
			delete [] node->LinkedList[i];
		}
	}
	delete [] leftChild->lowerbound;
	delete [] leftChild->upperbound;
	delete leftChild;
	leftChild = NULL;
	if (rightChild != NULL) {
		stream(rightChild);
		delete [] rightChild->lowerbound;
		delete [] rightChild->upperbound;
	}
	delete rightChild;
	rightChild = NULL;
	delete [] node->partialCoverIndex;
}

int main(int  argc, char  *argv[]) {

	int i, j, k, l;
	int popsize_original;

	/* check parameters */
	if (argc < 6)  {
		fprintf(stderr, "usage: gHSSP <number of points> <dimension> <number of points to be reserved> <input file> <reference point file> <outputfile(optional)>\n");
		exit(1);
	}
	sscanf(argv[1], "%d", &popsize_original);
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
	vector<double*>population_original = vector<double*>(popsize_original);
	population = vector<double*>(popsize_original);
	for (i=0; i<popsize_original; i++) {
		population_original[i] = new double[dimension];
		population[i] = new double[dimension];
		for (j=0; j<dimension; j++) {
			fileData >> word;
			population_original[i][j] = ref[j] - atof(word);
		}
	}
	fileData.close();

	struct timeval tv1, tv2;
	struct rusage ru_before, ru_after;
	
	getrusage (RUSAGE_SELF, &ru_before);
	
	int *sorted_height = new int[popsize_original];
	for (i=0; i<popsize_original; i++) {
		sorted_height[i] = i;
	}
	Index_Descend_Sort(population_original, sorted_height, popsize_original, dimension-1);
	sort(population_original.begin(), population_original.end(), Yildiz_cmp);
	
	boundaries = new int[popsize_original];
	noBoundaries = new int[popsize_original];
	projection = new bool[popsize_original];
	piles = new int[popsize_original];
	A0 = new int[dimension-1];
	A1 = new int[dimension-1];
	contributions = new double[popsize_original];
	
	vector<bool> deleted = vector<bool>(popsize_original, false);
	int count;
	int index = -1;
	double minContribution = 1e30;
	for (int i=0; i<popsize_original-k; i++) {
		popsize = popsize_original - i;
		// sqrt of popsize
		dSqrtDataNumber = sqrt((double)popsize);
		count = 0;
		for (j=0; j<popsize_original; j++) {
			if (!deleted[j]) {
				for (l=0; l<dimension; l++) {
					population[count][l] = population_original[j][l];
				}
				contributions[count] = 0.;
				count++;
			}
		}
		BFTreeNode *root = new BFTreeNode();
		root->split = 0;
		root->partialCoverNumber = popsize;
		root->partialCoverIndex = new int[popsize];
		root->lowerbound = new double[dimension-1];
		root->upperbound = new double[dimension-1];
		root->dims = vector<int>(dimension-1);
		for (j=0; j<popsize; j++) {
			root->partialCoverIndex[j] = j;
		}
		root->fullyCoveringBox.reserve(2);
		root->isNewLinkedList = true;
		root->LinkedList = vector<int*>(dimension-1);
		for (j=0; j<dimension-1; j++) {
			root->LinkedList[j] = new int[popsize];
			for (l=0; l<popsize; l++) {
				root->LinkedList[j][l] = l;
			}
			Index_Ascend_Sort(population, root->LinkedList[j], popsize, j);
		}
		for (j=0; j<dimension-1; j++) {
			root->lowerbound[j] = 0.;
			root->upperbound[j] = population[root->LinkedList[j][popsize-1]][j];
			root->dims[j] = j;
		}
		stream(root);
		delete [] root->lowerbound;
		delete [] root->upperbound;
		delete root;
		
		count = 0;
		index = -1;
		minContribution = 1e30;
		for (j=0; j<popsize_original; j++) {
			if (deleted[j]) {
				continue;
			}			
			if (contributions[count] < minContribution) {
				index = j;
				minContribution = contributions[count];
			}
			count++;
		}
		deleted[index] = true;
	}
	
	// find indices of remaining points in the original order
	vector<bool> selected = vector<bool>(popsize_original, false);
	for (i=0; i<popsize_original; i++) {
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
		for (i=0; i<popsize_original; i++) {
			if (selected[i]) {
				// Note: index starts from 0
				myoutput << i << endl;
			}
		}
		// the last line outputs the running time
		myoutput << setprecision(8) << tv2.tv_sec + tv2.tv_usec * 1e-6 - tv1.tv_sec - tv1.tv_usec * 1e-6;
		myoutput.close();
	} else {
		for (i=0; i<popsize_original; i++) {
			if (selected[i]) {
				printf("%d\n", i);
			}
		}
		printf("Time(s): %.10g\n", tv2.tv_sec + tv2.tv_usec * 1e-6 - tv1.tv_sec - tv1.tv_usec * 1e-6);
		printf("memory peak: %.1fMB\n", ru_after.ru_maxrss/(double) 1000);
	}

	return 0;
}
