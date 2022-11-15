# Introduction
A project aimed at parallelizing the convex hull algorithms using different tools.

# Algorithm 1 (Jarvis March)
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

## Algorithm 1 (Jarvis March) Using Open MPI
<img src="https://user-images.githubusercontent.com/70642284/201856798-294e2084-85f8-4b7c-95ab-961c579f2080.jpg" alt="result_open_mpi" width = "500" height = "540"> <img src="https://user-images.githubusercontent.com/70642284/201845608-65b9a431-a3df-44e2-acc7-d7f804dc756f.jpg" alt="result_open_mpi" width = "480" height = "400">

## Algorithm 1 (Jarvis March) Using Open MP
<img src="https://user-images.githubusercontent.com/70642284/201857356-8a7b0c4c-faca-48ce-a146-17b0c0294230.jpg" alt="result_open_mpi" width = "500" height = "540">


# Algorithm 2 (Jarvis March)
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
