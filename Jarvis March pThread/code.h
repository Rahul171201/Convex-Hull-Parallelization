#include <bits/stdc++.h>
#include <pthread.h>

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
    int count1 = 0;
    int p = starting_point, q;
    do
    {
        // Add current point to the convex hull
        hull_x[count1] = x[p];
        hull_y[count1] = y[p];
        count1++;
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
    //cout << count1 << endl;
    delete []hull_x;
    delete []hull_y;
}

int *hull_x, *hull_y, *X, *Y, *x, *y, n;
int starting_point;
int count1;
int p, q;
int n_threads;
int thread_min[NUM_THREADS];
int partition_size;
pthread_barrier_t barrier;

void *parallel_reducer(void *threadarg)
{
    long tid = (long)threadarg;
    if (tid == 0)
    {   
        starting_point = getStartingPoint(x, y, n);
        count1 = 0;
        p = starting_point;

        n_threads = NUM_THREADS;
        partition_size = n / n_threads;
        if (n % n_threads != 0)
        {
            partition_size++;
        }
        // cout << "T:" << n_threads << endl;
    }

    pthread_barrier_wait(&barrier);

    do
    {
        if (tid == 0)
        {
            // Add current point to the convex hull
            hull_x[count1] = x[p];
            hull_y[count1] = y[p];
            count1++;
            q = (p + 1) % n; // Let's say q is the most counter clockwise point
        }

        pthread_barrier_wait(&barrier);

        int local_q = q;
        for (int i = tid * partition_size; i < min((long int)n, (tid + 1) * partition_size); i++)
        {
            if (orientation(x[p], y[p], x[i], y[i], x[local_q], y[local_q]) == 2)
                local_q = i;
        }
        thread_min[tid] = local_q;

        pthread_barrier_wait(&barrier);

        if (tid == 0)
        {
            q = thread_min[0];
            for (int i = 1; i < n_threads; i++)
            {
                if (orientation(x[p], y[p], x[thread_min[i]], y[thread_min[i]], x[q], y[q]) == 2)
                    q = thread_min[i];
            }
            p = q;
        }
        pthread_barrier_wait(&barrier);
    } while (p != starting_point); // Repeat the process until we again reach the starting point
}


// Main function
int main(int argc, char *argv[])
{

    for (auto NN : {1e3, 1e4, 1e5, 1e6, 1e7})
    {
        for (auto max_val : {1e1, 1e2, 1e3, 1e4})
        {
            N = NN;
            MAX_VAL = max_val;

            long double serial_time, parallel_time;

            srand(time(0));

            X = new int[N];
            Y = new int[N];

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

            x = X;
            y = Y;
            n = N;

            start = high_resolution_clock::now();

            // There must be at least 3 points for a convex hull to be possible
            if (n < 3)
            {
                cout << "Convex Hull cannot be formed\n";
                exit(0);
            }

            hull_x = new int[n]; // stores the convex hull points
            hull_y = new int[n];

            int failed;
            int current_thread;
            pthread_t threads[NUM_THREADS];
            pthread_attr_t attr;
            void *status;

            pthread_barrier_init (&barrier, NULL, NUM_THREADS);
            pthread_attr_init(&attr);
            pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

            for (current_thread = 0; current_thread < NUM_THREADS; current_thread++)
            {
                // Create threads
                failed = pthread_create(&threads[current_thread], &attr, parallel_reducer, (void *)current_thread);
                if (failed)
                {
                    cout << "Error : Unable to create thread " << failed << endl;
                    exit(-1);
                }
            }

            pthread_attr_destroy(&attr);
            for (current_thread = 0; current_thread < NUM_THREADS; current_thread++)
            {
                failed = pthread_join(threads[current_thread], &status);
                if (failed)
                {
                    cout << "Error : Unable to join thread " << failed << endl;
                    exit(-1);
                }
            }
            stop = high_resolution_clock::now();
            // cout << count1 << endl;

            // // Get duration. Substart timepoints to
            // // get duration. To cast it to proper unit
            // // use duration cast method
            duration = duration_cast<microseconds>(stop - start);

            // cout << "Time taken by OpenMPI Jarvis March Algorithm = " << duration.count() / 1e6 << " microseconds" << endl;

            parallel_time = duration.count();

            // int convex_hull[2 * count];
            // for (int i = 0; i < count; i = i + 1)
            // {
            //     convex_hull[2 * i] = hull_x[i];
            //     convex_hull[2 * i + 1] = hull_y[i];
            // }

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

            delete []X;
            delete []Y;
            delete []hull_x;
            delete []hull_y;

            serial_time /= 1e6;
            parallel_time /= 1e6;

            cout << "N:" << N << ", MAX_VAL:" << MAX_VAL << ", Serial time:" << serial_time << ", Parallel Time with pthread:" << parallel_time << ", Speedup:" << serial_time / parallel_time << endl;

            // cout << fixed << serial_time << "," << parallel_time << "," << serial_time / parallel_time << endl;
        }
    }

    return 0;
}

// RESULTS
// N:1000, MAX_VAL:10, Serial time:8.5e-05, Parallel Time with pthread:0.003826, Speedup:0.0222164
// N:1000, MAX_VAL:100, Serial time:0.000117, Parallel Time with pthread:0.005396, Speedup:0.0216827
// N:1000, MAX_VAL:1000, Serial time:0.000119, Parallel Time with pthread:0.009547, Speedup:0.0124646
// N:1000, MAX_VAL:10000, Serial time:0.000131, Parallel Time with pthread:0.006809, Speedup:0.0192392
// N:10000, MAX_VAL:10, Serial time:0.000777, Parallel Time with pthread:0.006438, Speedup:0.12069
// N:10000, MAX_VAL:100, Serial time:0.00149, Parallel Time with pthread:0.008811, Speedup:0.169107
// N:10000, MAX_VAL:1000, Serial time:0.002585, Parallel Time with pthread:0.013187, Speedup:0.196026
// N:10000, MAX_VAL:10000, Serial time:0.001631, Parallel Time with pthread:0.009263, Speedup:0.176077
// N:100000, MAX_VAL:10, Serial time:0.008445, Parallel Time with pthread:0.006693, Speedup:1.26177
// N:100000, MAX_VAL:100, Serial time:0.014201, Parallel Time with pthread:0.0128, Speedup:1.10945
// N:100000, MAX_VAL:1000, Serial time:0.025496, Parallel Time with pthread:0.02164, Speedup:1.17819
// N:100000, MAX_VAL:10000, Serial time:0.01959, Parallel Time with pthread:0.019087, Speedup:1.02635
// N:1000000, MAX_VAL:10, Serial time:0.083197, Parallel Time with pthread:0.035998, Speedup:2.31116
// N:1000000, MAX_VAL:100, Serial time:0.143714, Parallel Time with pthread:0.069092, Speedup:2.08004
// N:1000000, MAX_VAL:1000, Serial time:0.254966, Parallel Time with pthread:0.119265, Speedup:2.13781
// N:1000000, MAX_VAL:10000, Serial time:0.19614, Parallel Time with pthread:0.090657, Speedup:2.16354
// N:10000000, MAX_VAL:10, Serial time:0.892006, Parallel Time with pthread:0.354567, Speedup:2.51576
// N:10000000, MAX_VAL:100, Serial time:1.46146, Parallel Time with pthread:0.554043, Speedup:2.6378
// N:10000000, MAX_VAL:1000, Serial time:2.79108, Parallel Time with pthread:1.07982, Speedup:2.58475
// N:10000000, MAX_VAL:10000, Serial time:2.39218, Parallel Time with pthread:0.890516, Speedup:2.68628