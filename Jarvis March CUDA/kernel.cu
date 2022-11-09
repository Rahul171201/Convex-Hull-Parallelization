
#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <chrono>
#include <fstream>

using namespace std;
using namespace std::chrono;

#define N 64      // number of points in the plane
#define MAX_VAL 20 // maximum value of any point in the plane

int bottom_most_point_x = 0;
int bottom_most_point_y = 0;

// To find orientation of ordered triplet (p, q, r).
// The function returns following values
// 0 --> p, q and r are collinear
// 1 --> Clockwise
// 2 --> Counterclockwise
// device function
__device__ int orientation(int p_x, int p_y, int q_x, int q_y, int r_x, int r_y)
{
    int val = (q_y - p_y) * (r_x - q_x) - (q_x - p_x) * (r_y - q_y);

    if (val == 0)
        return 0;             // collinear
    return (val > 0) ? 1 : 2; // clock or counterclock wise
}

int getStartingPoint(int *x, int *y, int n)
{
    int min_y = INT_MAX, min_x = INT_MAX;
    int index = -1;
    for (int i = 0; i < n; i++)
    {
        if (y[i] < min_y)
        {
            min_y = y[i];
            min_x = x[i];
            index = i;
        }
        else if (y[i] == min_y)
        {
            if (min_x > x[i])
                index = i;
            min_x = min(min_x, x[i]);
        }
    }
    return index;
}

__global__ void min_angle(int *x, int *y, int p, int n, int *temp_point)
{
    int i = threadIdx.x;
    int stride = 1;
    int temp_rem = 2;
    //printf("Im thread number %d\n", i);

    if (i >= n)
        return;

    __syncthreads();

    while (stride < n) {
        
        if (i % temp_rem == 0) {
            
            if (orientation(x[p], y[p], x[temp_point[i+stride]], y[temp_point[i+stride]], x[temp_point[i]], y[temp_point[i]]) == 2) {
                temp_point[i] = temp_point[i+stride];
            }
            //printf("thread  number = %d has temp_point = %d\n", i, temp_point[i]);
        }
        
        temp_rem = temp_rem *2;
        stride = stride * 2;
        
        __syncthreads();
    }
}

// Main function
int main()
{
    srand(time(0));
    
    int* x;
    int* y;

    cudaMallocManaged(&x, N * sizeof(int));
    cudaMallocManaged(&y, N * sizeof(int));

    for (int i = 0; i < N; i++)
    {
        int val_x = (rand() % (MAX_VAL - 1 + 1)) + 1;
        int val_y = (rand() % (MAX_VAL - 1 + 1)) + 1;
        x[i] = val_x;
        y[i] = val_y;
    }

    // Convex Hull Computation

    // There must be at least 3 points
    if (N < 3)
    {
        cout << "Convex hull is not possible, since we require more than 3 points\n";
        exit(0);
    }

    // Initialize Result
    int* hull_x = (int*)malloc(N * sizeof(int));
    int* hull_y = (int*)malloc(N * sizeof(int));

    int* temp_point;

    cudaMallocManaged(&temp_point, N * sizeof(int));

    // Find the leftmost point
    int starting_point = getStartingPoint(x, y, N);

    // Start from leftmost point, keep moving counterclockwise
    // until reach the start point again.  This loop runs O(h)
    // times where h is number of points in result or output.
    int p = starting_point;
    int count = 0;
    do
    {
        //cout << "value of p = " << p << "\n";
        // Add current point to result
        hull_x[count] = x[p];
        hull_y[count] = y[p];
        count++;

        for (int i = 0; i < N; i++) {
            temp_point[i] = i;
        }

        // Search for a point 'q' such that orientation(p, q,
        // x) is counterclockwise for all points 'x'. The idea
        // is to keep track of last visited most counterclock-
        // wise point in q. If any point 'i' is more counterclock-
        // wise than q, then update q.

        min_angle << <1, N >> > (x, y, p, N, temp_point);
        cudaDeviceSynchronize();

        

        // Now q is the most counterclockwise with respect to p
        // Set p as q for next iteration, so that q is added to
        // result 'hull'
        p = temp_point[0];
        //cout << "value of p = " << p << "\n";
    } while (p != starting_point); // While we don't come to first point

    cudaFree(temp_point);


    int* convex_hull = (int*)malloc(2 * N * sizeof(int));

    for (int i = 0; i < N; i++) {
        convex_hull[2 * i] = hull_x[i];
        convex_hull[(2 * i + 1)] = hull_y[i];
    }

    cout << "The points are:\n";
    for (int i=0;i<N;i++)
    {
        cout << "{" << x[i] << ", " << y[i] << "}\n";
    }

    cout << "The convex hull points are:\n";
    for (int i = 0; i < 2*count; i=i+2) {
        cout << convex_hull[i] << " " << convex_hull[i+1] << "\n";
    }

    freopen("points.txt", "w", stdout);

    cout << N << "\n";

    for (int i=0;i<N;i++)
    {
        cout << x[i] << " " << y[i] << "\n";
    }

    cout << count << "\n";

    for (int i=0;i< 2*count;i=i+2)
    {
        cout << convex_hull[i] << " " << convex_hull[i+1] << "\n";
    }
    
}