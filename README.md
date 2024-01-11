# gHSSD-Tree
(see gHSSDTree.cpp)
An efficient algorithm for greedy decremental hypervolume subset selection (gHSSD)

The time complexity is $O((n-k+\sqrt{n})n^{\frac{d-1}{2}}\log n)$ where n is the number of points, d is the dimensionality, and k is the number of points to be reserved.

This is the implementation of our method proposed in:

Jingda Deng, Jianyong Sun, Qingfu Zhang, and Hui Li, "Efficient Greedy Decremental Hypervolume Subset Selection
Using Space Partition Tree" (submitted to IEEE-TEVC)

# gHSSD by BF
(see gHSSDbyBF.cpp)

Our implementation for the Bringmann and Friedrich's algorithm [1], and its application for gHSSD.

The time complexity is $O((n-k)n^{\frac{d}{2}}\log n)$

[1] Karl Bringmann, Tobias Friedrich, "An Efficient Algorithm for Computing Hypervolume Contributions", Evolutionary Computation, vol. 18, no. 3, pp. 383-402, 2010.

# Test Set
Test sets in the numerical experiments including:
  - Random sets: spherical, cliff, linear sets
  - Point sets from EMOA: DTLZ1-DTLZ7 solution sets

# Contact
Jingda Deng

School of Mathematics and Statistics

Xi'an Jiaotong University

E-mail: jddeng@xjtu.edu.cn

# Update Log
2024/01/12 Improve the implementation of gHSSD-Tree, with 20%-50% speed-up
