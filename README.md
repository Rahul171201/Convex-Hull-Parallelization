# Introduction
Convex hull problem corresponds to the task of finding out the smallest convex set that encloses a given set of points. The points can be located in 2D, 3D or even higher dimensions. For the sake of simplicity, we’ll limit our scope of project to points given in a 2D space.

Convex hull problem has a wide variety of applications ranging from linear algebra to graphical computing. Convex hull is widely used in robot motion planning to find out the optimal path of a robot avoiding obstacles. Similar to robot path planning, It is also used in path planning of autonomous vehicles. 

The real-time computations involving the use of convex hulls in various fields like computer graphics and image processing get very challenging once the number of points cross 106. In our project, we will discover and analyze various algorithms that can be used to help us parallelize the process of finding out the convex hull of a given set of points.

<img src="https://user-images.githubusercontent.com/70642284/201902932-3e1a808c-3b69-4138-a3e7-b079740b4394.png" alt="result_open_mpi" width = "400" height = "340">


# Sequential Jarvis March Algorithm
* After distribution of points in the plane, insert all the points in an array.
* Find the bottom-most point in the plane by iterating over all the points sequentially one by one and checking if it has a smaller y coordinate than the current bottommost point.
* The bottommost point will be our current starting point.
* Iterate over all the points sequentially and find out the least counter clockwise point with respect to the current point. The least counter clockwise point will become the starting point for the next iteration and is pushed to the convex hull.
*  The process is repeated until we reach the bottommost point again and the convex hull is hence calculated.


## Overall Complexity:
```
The bottommost point is calculated in ‘n’ steps using sequential search.
Then the calculation of minimum counter clockwise point is done ‘h’ times, where h is the number of points in the convex hull.
Computation of minimum counter clockwise points is done in n steps again using parallel reduction technique.
So overall complexity = n + h * n
Hence final complexity ~ nh

```


# Parallel Algorithm 1 (Jarvis March)
* After distribution of points in the plane, insert all the points in an array.
* Divide the array into k partitions, where k is the number of threads.
* Find the bottom-most point in the plane. It will be our starting point in the algorithm.
* Each thread works on its respective partition and calculates the least counter clockwise angle that is being made by any point in its respective partition.
* After all threads have calculated their respective least counter clockwise angle, the results from all the threads are combined and the final lowest counter clockwise point is computed. 
* This calculated point becomes the starting point for the next iteration. The point is pushed into our convex hull.
* The process is repeated until we reach the starting point again and the convex hull is hence calculated.

## Overall Complexity:
```
The calculation of minimum counter clockwise point is done ‘h’ times, where h is the number of points in the convex hull.
Each thread carries out n/p number of computations where n is the total number of elements and p is the number of processes.
Finally combining the results from all the threads takes p steps. So overall complexity = h[ (n/p) + p ]. 
Considering n>>p, we can say that the overall complexity ~ O(nh/p).
```

## Results of Parallel Algorithm 1 (Jarvis March) Using Open MPI
<img src="https://user-images.githubusercontent.com/70642284/201856798-294e2084-85f8-4b7c-95ab-961c579f2080.jpg" alt="result_open_mpi" width = "500" height = "540"> <img src="https://user-images.githubusercontent.com/70642284/201845608-65b9a431-a3df-44e2-acc7-d7f804dc756f.jpg" alt="result_open_mpi" width = "480" height = "400">

## Results of Parallel Algorithm 1 (Jarvis March) Using Open MP
<img src="https://user-images.githubusercontent.com/70642284/201857356-8a7b0c4c-faca-48ce-a146-17b0c0294230.jpg" alt="result_open_mpi" width = "500" height = "540">

## Results of Parallel Algorithm 1 (Jarvis March) Using pthread
<img src="https://user-images.githubusercontent.com/70642284/201902370-d55d0041-37fa-4ad1-83de-1554e522e558.png" alt="result_open_mpi" width = "500" height = "540">


# Parallel Algorithm 2 (Jarvis March)
* After distribution of points in the plane, insert all the points in an array.
* Calculate the bottommost starting point using parallel reduction technique. This will be in O(log(n)).
* Now we assign each element of the array to a thread. Each thread stores the least counter clockwise point (initially itself) in a res array.
* Then using parallel reduction, the least counter clockwise point is calculated and stored in the first index of the res array. This also takes O(log(n)).
* This calculated point becomes the starting point for the next iteration. The point is pushed into our convex hull.
* The process is repeated until we reach the starting point again and the convex hull is hence calculated.

## Overall Complexity:
```
The bottommost point is calculated in log(n) steps using parallel reduction. 
Then the calculation of minimum counter clockwise point is done ‘h’ times, where h is the number of points in the convex hull. 
Computation of minimum counter clockwise point is done in log(n) steps again using parallel reduction technique. 
So overall complexity = log(n) + h[ log(n) ]
Hence final complexity ~ hlog(n)
```

#Conclusion
In our report, we use the metric of speedup to benchmark and compare the parallel implementations of algorithms. We plot the graph of speedup against the size of the input (i.e. number of given points). As observed, for small inputs, up until (N < 105), the serial versions perform better than their parallelized counterparts (because speedup < 1). As the input size gets larger, the speedup drastically increases. For large values of N (in the order of 107), all the parallelized algorithms are almost 2-3 times faster their serial counterparts, if not more (because speedup > 2).

# References
* Convex Hull | Set 1 (Jarvis’s Algorithm or Wrapping) : https://www.geeksforgeeks.org/convex-hull-set-1-jarviss-algorithm-or-wrapping/

* A BSP realization of Jarvis's algorithm : https://ieeexplore.ieee.org/document/797603

* Convex Hull | Set 2 (Graham Scan) : https://www.geeksforgeeks.org/convex-hull-set-2-graham-scan/
