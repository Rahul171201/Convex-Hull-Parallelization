#include <bits/stdc++.h>
#include <omp.h>

using namespace std;
using namespace std::chrono;

// #define N 15      // number of points in the plane
// #define MAX_VAL 1000 // maximum value of any point in the plane
#define NUM_THREADS 8

int N = 10, MAX_VAL = 100;

// Function to find orientation of points p,q and r
// 0 --> p, q and r are collinear
// 1 --> Clockwise
// 2 --> Counterclockwise
int orientation(int p_x, int p_y, int q_x, int q_y, int r_x, int r_y)
{
    int val = (q_y - p_y) * (r_x - q_x) - (q_x - p_x) * (r_y - q_y);

    if (val == 0)
        return 0;
    return (val > 0) ? 1 : 2;
}

// Function to get the bottom most point
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

// Function that defines the Jarvis March algorithm
void sequentialJarvisMarch(int *x, int *y, int n)
{
     // There must be at least 3 points for a convex hull to be possible
    if (n < 3)
    {
        cout << "Convex Hull cannot be formed\n";
        exit(0);
    }

    int *hull_x = new int[n]; // stores the convex hull points
    int *hull_y = new int[n];

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
            if (orientation(x[p], y[p], x[i], y[i], x[q], y[q]) == 2)
                q = i;
        }
        p = q;

    } while (p != starting_point); // Repeat the process until we again reach the starting point

    // cout<<"The convex hull points after sequential algoroithm are:\n";
    // for(int i=0;i<n;i++){
    //     cout<<hull_x[i]<<" "<<hull_y[i]<<"\n";
    // }
    // cout << count << endl;
    delete []hull_x;
    delete []hull_y;
}

int *hull_x, *hull_y;

// Main function
int main(int argc, char *argv[])
{

    for(auto NN:{1e3, 1e4, 1e5, 1e6, 1e7}){
        for(auto max_val:{1e1, 1e2, 1e3, 1e4}){
            N = NN;
            MAX_VAL = max_val;

            long double serial_time, parallel_time;

            srand(time(0));

            int *X = new int[N], *Y = new int[N];

            for (int i = 0; i < N; i++)
            {
                int val_X = (rand() % (MAX_VAL - 1 + 1)) + 1;
                int val_Y = (rand() % (MAX_VAL - 1 + 1)) + 1;
                X[i] = val_X;
                Y[i] = val_Y;
            }

            auto start = high_resolution_clock::now();
            sequentialJarvisMarch(X, Y, N);
            auto stop = high_resolution_clock::now();
            auto duration = duration_cast<microseconds>(stop - start);

            // cout << "Time taken by sequential algorithm = " << duration.count() / 1e6 << " microseconds" << endl;

            serial_time = duration.count();

            int *x = X, *y = Y;
            int n = N;

            start = high_resolution_clock::now();
            

            // There must be at least 3 points for a convex hull to be possible
            if (n < 3)
            {
                cout << "Convex Hull cannot be formed\n";
                exit(0);
            }

            hull_x = new int[n]; // stores the convex hull points
            hull_y = new int[n];

            // Find the bottommost point
            int starting_point = getStartingPoint(x, y, n);
            int count = 0;
            int p = starting_point, q;
            omp_set_num_threads(NUM_THREADS);
            int n_threads;
            int thread_min[NUM_THREADS];
            int partition_size;

            #pragma omp parallel
            {
                int tid = omp_get_thread_num();
                if (tid == 0)
                {
                    n_threads = omp_get_num_threads();
                    partition_size = n / n_threads;
                    if(n % n_threads!=0){
                        partition_size++;
                    }
                    // cout << "T:" << n_threads << endl;
                }

                #pragma omp barrier

                do
                {   
                    if(tid == 0){
                        // Add current point to the convex hull
                        hull_x[count] = x[p];
                        hull_y[count] = y[p];
                        count++;
                        q = (p + 1) % n; // Let's say q is the most counter clockwise point
                    }

                    #pragma omp barrier

                    int local_q = q;
                    for (int i = tid * partition_size; i < min(n, (tid+1)*partition_size); i++){
                        if (orientation(x[p], y[p], x[i], y[i], x[local_q], y[local_q]) == 2)
                            local_q = i;
                    }
                    thread_min[tid] = local_q;

                    #pragma omp barrier

                    if (tid == 0)
                    {
                        q = thread_min[0];
                        for (int i = 1; i < n_threads; i++){
                            if (orientation(x[p], y[p], x[thread_min[i]], y[thread_min[i]], x[q], y[q]) == 2)
                                q = thread_min[i];
                        }
                        p = q;
                    }
                    #pragma omp barrier
                } while (p != starting_point); // Repeat the process until we again reach the starting point        
            }
            stop = high_resolution_clock::now();
            // cout << count << endl;

            // // Get duration. Substart timepoints to
            // // get duration. To cast it to proper unit
            // // use duration cast method
            duration = duration_cast<microseconds>(stop - start);

            // cout << "Time taken by OpenMPI Jarvis March Algorithm = " << duration.count() / 1e6 << " microseconds" << endl;

            parallel_time = duration.count();

            int convex_hull[2 * count];
            for (int i = 0; i < count; i = i + 1)
            {
                convex_hull[2 * i] = hull_x[i];
                convex_hull[2 * i + 1] = hull_y[i];
            }

            // cout << "The points in the convex hull are\n";
            // for (int i = 0; i < 2 * count; i = i + 2)
            // {
            //     cout << convex_hull[i] << " " << convex_hull[i + 1] << "\n";
            // }

            // freopen("points.txt", "w", stdout);
            // cout << N << "\n";
            // for (int i = 0; i < N; i++)
            // {
            //     cout << x[i] << " " << y[i] << "\n";
            // }

            // cout << count << "\n";
            // for (int i = 0; i < 2 * count; i = i + 2)
            // {
            //     cout << convex_hull[i] << " " << convex_hull[i + 1] << "\n";
            // }

            delete X;
            delete Y;
            delete hull_x;
            delete hull_y;

            serial_time /= 1e6;
            parallel_time /= 1e6;


            cout << "N:" << N << ", MAX_VAL:" << MAX_VAL << ", Serial time:" << serial_time << ", Parallel Time with OpenMP:" << parallel_time << ", Speedup:" << serial_time / parallel_time << endl;
            
            // cout << fixed<< serial_time << "," << parallel_time << "," << serial_time / parallel_time << endl;
        }
    }

    
    return 0;
}

// RESULTS
// N:1000, MAX_VAL:10, Serial time:0.000178, Parallel Time with OpenMP:0.005187, Speedup:0.0343166
// N:1000, MAX_VAL:100, Serial time:0.00014, Parallel Time with OpenMP:0.007155, Speedup:0.0195667
// N:1000, MAX_VAL:1000, Serial time:0.000119, Parallel Time with OpenMP:0.006641, Speedup:0.017919
// N:1000, MAX_VAL:10000, Serial time:0.000133, Parallel Time with OpenMP:0.009094, Speedup:0.014625
// N:10000, MAX_VAL:10, Serial time:0.001187, Parallel Time with OpenMP:0.006325, Speedup:0.187668
// N:10000, MAX_VAL:100, Serial time:0.001854, Parallel Time with OpenMP:0.009632, Speedup:0.192483
// N:10000, MAX_VAL:1000, Serial time:0.001492, Parallel Time with OpenMP:0.006913, Speedup:0.215825
// N:10000, MAX_VAL:10000, Serial time:0.00215, Parallel Time with OpenMP:0.011076, Speedup:0.194113
// N:100000, MAX_VAL:10, Serial time:0.012822, Parallel Time with OpenMP:0.024437, Speedup:0.524696
// N:100000, MAX_VAL:100, Serial time:0.022373, Parallel Time with OpenMP:0.025805, Speedup:0.867003
// N:100000, MAX_VAL:1000, Serial time:0.016313, Parallel Time with OpenMP:0.019976, Speedup:0.81663
// N:100000, MAX_VAL:10000, Serial time:0.028948, Parallel Time with OpenMP:0.037852, Speedup:0.764768
// N:1000000, MAX_VAL:10, Serial time:0.119412, Parallel Time with OpenMP:0.063419, Speedup:1.88291
// N:1000000, MAX_VAL:100, Serial time:0.118673, Parallel Time with OpenMP:0.060253, Speedup:1.96958
// N:1000000, MAX_VAL:1000, Serial time:0.237947, Parallel Time with OpenMP:0.114205, Speedup:2.08351
// N:1000000, MAX_VAL:10000, Serial time:0.262057, Parallel Time with OpenMP:0.154009, Speedup:1.70157
// N:10000000, MAX_VAL:10, Serial time:1.27074, Parallel Time with OpenMP:0.481439, Speedup:2.63947
// N:10000000, MAX_VAL:100, Serial time:1.98778, Parallel Time with OpenMP:0.760886, Speedup:2.61245
// N:10000000, MAX_VAL:1000, Serial time:1.75331, Parallel Time with OpenMP:0.657579, Speedup:2.66631
// N:10000000, MAX_VAL:10000, Serial time:3.15807, Parallel Time with OpenMP:1.21991, Speedup:2.58878