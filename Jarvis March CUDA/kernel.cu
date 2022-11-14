#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <chrono>
#include <fstream>

using namespace std;
using namespace std::chrono;

#define N 1024      // number of points in the plane
#define MAX_VAL 1000 // maximum value of any point in the plane

int bottom_most_point_x = 0;
int bottom_most_point_y = 0;

// Device function To find orientation of points p,q and r
// The function returns following values
// 0 --> p, q and r are collinear
// 1 --> Clockwise
// 2 --> Counterclockwise
__device__ int orientation(int p_x, int p_y, int q_x, int q_y, int r_x, int r_y)
{
    int val = (q_y - p_y) * (r_x - q_x) - (q_x - p_x) * (r_y - q_y);

    if (val == 0)
        return 0;
    return (val > 0) ? 1 : 2;
}

// Function to get the bottom most point
int getStartingPoint(int* x, int* y, int n)
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

// Kernel function to find the minimum counter clockwise angle
__global__ void min_angle(int* x, int* y, int p, int n, int* temp_point)
{
    int i = threadIdx.x; // get thread id
    int stride = 1; // stride
    int temp_rem = 2;

    if (i >= n)
        return;

    __syncthreads();

    while (stride < n) {
        if (i % temp_rem == 0) {
            if (orientation(x[p], y[p], x[temp_point[i + stride]], y[temp_point[i + stride]], x[temp_point[i]], y[temp_point[i]]) == 2) {
                temp_point[i] = temp_point[i + stride];
            }
        }

        temp_rem = temp_rem * 2;
        stride = stride * 2;

        __syncthreads(); // synchronization barrier
    }
}

int sequential_orientation(int p_x, int p_y, int q_x, int q_y, int r_x, int r_y)
{
    int val = (q_y - p_y) * (r_x - q_x) - (q_x - p_x) * (r_y - q_y);

    if (val == 0)
        return 0;
    return (val > 0) ? 1 : 2;
}

// Function that defines the Jarvis March algorithm
void sequentialJarvisMarch(int* x, int* y, int n)
{
    // There must be at least 3 points for a convex hull to be possible
    if (n < 3)
    {
        cout << "Convex Hull cannot be formed\n";
        exit(0);
    }

    int* hull_x = (int*)malloc(n * sizeof(int)); // stores the convex hull points
    int* hull_y = (int*)malloc(n * sizeof(int));

    // Find the bottommost point
    int starting_point = getStartingPoint(x, y, n);
    int count = 0;
    int p = starting_point, q;
    do
    {
        // Add current point to the convex hull
        hull_x[count] = x[p];
        hull_y[count] = y[p];
        count++;
        q = (p + 1) % n; // Let's say q is the most counter clockwise point

        for (int i = 0; i < n; i++)
        {
            // If i is more counterclockwise than current q, then update q
            if (sequential_orientation(x[p], y[p], x[i], y[i], x[q], y[q]) == 2)
                q = i;
        }

        p = q;

    } while (p != starting_point); // Repeat the process until we again reach the starting point

    // cout<<"The convex hull points after sequential algoroithm are:\n";
    // for(int i=0;i<n;i++){
    //     cout<<hull_x[i]<<" "<<hull_y[i]<<"\n";
    // }
}

// Main function
int main()
{
    srand(time(0));

    int* x;
    int* y;

    // unified shared memory
    cudaMallocManaged(&x, N * sizeof(int));
    cudaMallocManaged(&y, N * sizeof(int));

    for (int i = 0; i < N; i++)
    {
        int val_x = (rand() % (MAX_VAL - 1 + 1)) + 1;
        int val_y = (rand() % (MAX_VAL - 1 + 1)) + 1;
        x[i] = val_x;
        y[i] = val_y;
    }

    // Sequential Convex Hull Computation
    auto start = high_resolution_clock::now();
    sequentialJarvisMarch(x, y, N);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);

    cout << "Time taken by sequential algorithm = " << duration.count() << " microseconds" << endl;

    // Parallel Convex Hull Computation

    // There must be at least 3 points for a convex hull to be possible
    if (N < 3)
    {
        cout << "Convex hull is not possible, since we require more than 3 points\n";
        exit(0);
    }

    // Initialize the hull
    int* hull_x = (int*)malloc(N * sizeof(int));
    int* hull_y = (int*)malloc(N * sizeof(int));

    int* temp_point;

    cudaMallocManaged(&temp_point, N * sizeof(int));

    // Find the bottommost starting point
    int starting_point = getStartingPoint(x, y, N);

    start = high_resolution_clock::now();
    int p = starting_point;
    int count = 0;
    do
    {
        // Add current point to convex hull
        hull_x[count] = x[p];
        hull_y[count] = y[p];
        count++;

        for (int i = 0; i < N; i++) {
            temp_point[i] = i;
        }

        min_angle << <1, N >> > (x, y, p, N, temp_point);
        cudaDeviceSynchronize();

        p = temp_point[0];

    } while (p != starting_point); // Repeat the process until and unless we reach ths starting point
    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop - start);

    cout << "Time taken by parallel algorithm = " << duration.count() << " microseconds" << endl;

    int* convex_hull = (int*)malloc(2 * N * sizeof(int));

    for (int i = 0; i < N; i++) {
        convex_hull[2 * i] = hull_x[i];
        convex_hull[(2 * i + 1)] = hull_y[i];
    }

    // cout << "The points are:\n";
    // for (int i = 0; i < N; i++)
    // {
    //     cout << "{" << x[i] << ", " << y[i] << "}\n";
    // }

    // cout << "The convex hull points are:\n";
    // for (int i = 0; i < 2 * count; i = i + 2) {
    //     cout << convex_hull[i] << " " << convex_hull[i + 1] << "\n";
    // }

    // write the output onto a file
    freopen("points.txt", "w", stdout);

    cout << N << "\n";

    for (int i = 0; i < N; i++)
    {
        cout << x[i] << " " << y[i] << "\n";
    }

    cout << count << "\n";

    for (int i = 0; i < 2 * count; i = i + 2)
    {
        cout << convex_hull[i] << " " << convex_hull[i + 1] << "\n";
    }

    // free memory
    free(hull_x);
    free(hull_y);
    cudaFree(temp_point);
    cudaFree(temp_point);
    cudaFree(temp_point);
}