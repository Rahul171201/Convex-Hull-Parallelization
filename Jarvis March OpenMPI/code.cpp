#include <bits/stdc++.h>
#include <mpi.h>

using namespace std;
using namespace std::chrono;

#define N 100000      // number of points in the plane
#define MAX_VAL 10000 // maximum value of any point in the plane

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
int getStartingPoint(int x[N], int y[N], int n)
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

    int *hull_x = (int *)malloc(n * sizeof(int)); // stores the convex hull points
    int *hull_y = (int *)malloc(n * sizeof(int));

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
}

// Main function
int main(int argc, char *argv[])
{

    int pid, np, elements_per_process, n_elements_recieved;
    // np -> no. of processes
    // pid -> process id

    MPI_Status status;

    // Creation of parallel processes
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    MPI_Comm_size(MPI_COMM_WORLD, &np);

    // Master process
    if (pid == 0)
    {
        srand(time(0));

        int X[N], Y[N];

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

        cout << "Time taken by sequential algorithm = " << duration.count() << " microseconds" << endl;

        int x[N], y[N];

        for (int i = 0; i < N; i++)
        {
            int val_x = (rand() % (MAX_VAL - 1 + 1)) + 1;
            int val_y = (rand() % (MAX_VAL - 1 + 1)) + 1;
            x[i] = val_x;
            y[i] = val_y;
        }

        // total number of points
        //cout << "The total number of points in the plane = " << N << "\n";

        //cout << "The points are\n";
        //for (int i = 0; i < N; i++)
        //{
        //    cout << x[i] << " " << y[i] << "\n";
        //}
        
        // Find the bottommost point
        int starting_point = getStartingPoint(x, y, N);
        
        // Check if number of processes is more than 1
        if (np > 1)
        {
            // There must be at least 3 points for a convex hull to be possible
            if (N < 3)
            {
                cout << "Convex hull is not possible, since we require more than 3 points\n";
                exit(0);
            }

            // Get starting timepoint
            start = high_resolution_clock::now();

            int hull_x[N];
            int hull_y[N];      

            int p = starting_point, q;
            int count = 0;

            // variable to check if message to be passed again
            int again = 1;

            do
            {
                for (int i = 1; i < np; i++)
                {
                    MPI_Send(&again, 1, MPI_INT, i, i - 1, MPI_COMM_WORLD);
                }

                // Add current point to the convex hull
                hull_x[count] = x[p];
                hull_y[count] = y[p];

                count++;

                int elements_per_process = (N / np);
                q = (p + 1) % N;

                // Message sent by master to all
                for (int i = 1; i < np; i++)
                {
                    int index = i * elements_per_process;

                    MPI_Send(&elements_per_process, 1, MPI_INT, i, i - 1, MPI_COMM_WORLD);
                    MPI_Send(&x[index], elements_per_process, MPI_INT, i, i - 1, MPI_COMM_WORLD);
                    MPI_Send(&y[index], elements_per_process, MPI_INT, i, i - 1, MPI_COMM_WORLD);
                    MPI_Send(&x[p], 1, MPI_INT, i, i - 1, MPI_COMM_WORLD);
                    MPI_Send(&y[p], 1, MPI_INT, i, i - 1, MPI_COMM_WORLD);
                }

                for (int i = 0; i < elements_per_process; i++)
                {
                    if (orientation(x[p], y[p], x[i], y[i], x[q], y[q]) == 2)
                        q = i;
                }

                // cout << "Message receving started by master\n";

                for (int i = 1; i < np; i++)
                {
                    int index;

                    MPI_Recv(&index, 1, MPI_INT, i, i - 1, MPI_COMM_WORLD, &status);
                    if (orientation(x[p], y[p], x[index], y[index], x[q], y[q]) == 2)
                        q = index;
                }

                // Now q is the most counterclockwise with respect to p
                // Set p as q for next iteration, so that q is added to
                // result 'hull'
                p = q;

            } while (p != starting_point); // While we don't come to first point

            again = 0;
            for (int i = 1; i < np; i++)
            {
                MPI_Send(&again, 1, MPI_INT, i, i - 1, MPI_COMM_WORLD);
            }

            // Get ending timepoint
            stop = high_resolution_clock::now();

            // Get duration. Substart timepoints to
            // get duration. To cast it to proper unit
            // use duration cast method
            duration = duration_cast<microseconds>(stop - start);

            cout << "Time taken by OpenMPI Jarvis March Algorithm = " << duration.count() << " microseconds" << endl;

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
        }
    }
    // slave processes
    else
    {

        int again = 1;
        MPI_Recv(&again, 1, MPI_INT, 0, pid - 1, MPI_COMM_WORLD, &status);
        while (again)
        {
            // cout << "Message receving started by "<<pid<<"\n";
            // stores the received array segment
            // in local array a2
            MPI_Recv(&n_elements_recieved, 1, MPI_INT, 0, pid - 1, MPI_COMM_WORLD, &status);

            int arr_x[n_elements_recieved];
            int arr_y[n_elements_recieved];
            int p_x, p_y;

            MPI_Recv(&arr_x, n_elements_recieved, MPI_INT, 0, pid - 1, MPI_COMM_WORLD, &status);
            MPI_Recv(&arr_y, n_elements_recieved, MPI_INT, 0, pid - 1, MPI_COMM_WORLD, &status);
            MPI_Recv(&p_x, 1, MPI_INT, 0, pid - 1, MPI_COMM_WORLD, &status);
            MPI_Recv(&p_y, 1, MPI_INT, 0, pid - 1, MPI_COMM_WORLD, &status);

            // calculates its partial sum
            int q = 0;

            for (int i = 0; i < n_elements_recieved; i++)
                if (orientation(p_x, p_y, arr_x[i], arr_y[i], arr_x[q], arr_y[q]) == 2)
                    q = i;

            int final_index = pid * n_elements_recieved + q;
            // sends the partial sum to the root process

            // cout << "Message sending started by "<<pid<<" with the value = "<<final_index<<"\n";
            MPI_Send(&final_index, 1, MPI_INT, 0, pid - 1, MPI_COMM_WORLD);

            MPI_Recv(&again, 1, MPI_INT, 0, pid - 1, MPI_COMM_WORLD, &status);
        }
    }

    // cleans up all MPI state before exit of process
    MPI_Finalize();

    return 0;
}

// RESULTS
// AFTER AVERAGING THE RESULTS OVER 5 ITERATIONS :

// FOR N = 10 AND MAX_VALUE = 10, TIME_BY_SERIAL_EXECUTION = 1 milliseconds, TIME_BY_PARALLEL_EXECUTION = 2131 milliseconds, RATIO(SERIAL/PARALLEL) = 0.000469 
// FOR N = 10 AND MAX_VALUE = 100, TIME_BY_SERIAL_EXECUTION = 1 milliseconds, TIME_BY_PARALLEL_EXECUTION = 7720.4 milliseconds, RATIO(SERIAL/PARALLEL) = 0.000129 
// FOR N = 10 AND MAX_VALUE = 1000, TIME_BY_SERIAL_EXECUTION = 1 milliseconds, TIME_BY_PARALLEL_EXECUTION = 5991.4 milliseconds, RATIO(SERIAL/PARALLEL) = 0.000166 
// FOR N = 10 AND MAX_VALUE = 10000, TIME_BY_SERIAL_EXECUTION = 1 milliseconds, TIME_BY_PARALLEL_EXECUTION = 4112 milliseconds, RATIO(SERIAL/PARALLEL) = 0.000243 

// FOR N = 100 AND MAX_VALUE = 10, TIME_BY_SERIAL_EXECUTION = 6.8 milliseconds, TIME_BY_PARALLEL_EXECUTION = 5578.6 milliseconds, RATIO(SERIAL/PARALLEL) = 0.001219 
// FOR N = 100 AND MAX_VALUE = 100, TIME_BY_SERIAL_EXECUTION = 7.8 milliseconds, TIME_BY_PARALLEL_EXECUTION = 1586.6 milliseconds, RATIO(SERIAL/PARALLEL) = 0.004916 
// FOR N = 100 AND MAX_VALUE = 1000, TIME_BY_SERIAL_EXECUTION = 7.6 milliseconds, TIME_BY_PARALLEL_EXECUTION = 2695.8 milliseconds, RATIO(SERIAL/PARALLEL) = 0.002819 
// FOR N = 100 AND MAX_VALUE = 10000, TIME_BY_SERIAL_EXECUTION = 7.2 milliseconds, TIME_BY_PARALLEL_EXECUTION = 3356 milliseconds, RATIO(SERIAL/PARALLEL) = 0.002145

// FOR N = 1000 AND MAX_VALUE = 10, TIME_BY_SERIAL_EXECUTION = 79.4 milliseconds, TIME_BY_PARALLEL_EXECUTION = 24656.4 milliseconds, RATIO(SERIAL/PARALLEL) = 0.003220 
// FOR N = 1000 AND MAX_VALUE = 100, TIME_BY_SERIAL_EXECUTION = 110.4 milliseconds, TIME_BY_PARALLEL_EXECUTION = 6311.4 milliseconds, RATIO(SERIAL/PARALLEL) = 0.017492 
// FOR N = 1000 AND MAX_VALUE = 1000, TIME_BY_SERIAL_EXECUTION = 114.4 milliseconds, TIME_BY_PARALLEL_EXECUTION = 1792.8 milliseconds, RATIO(SERIAL/PARALLEL) = 0.063810 
// FOR N = 1000 AND MAX_VALUE = 10000, TIME_BY_SERIAL_EXECUTION = 100 milliseconds, TIME_BY_PARALLEL_EXECUTION = 740.2 milliseconds, RATIO(SERIAL/PARALLEL) = 0.135098

// FOR N = 10000 AND MAX_VALUE = 10, TIME_BY_SERIAL_EXECUTION = 826.8 milliseconds, TIME_BY_PARALLEL_EXECUTION = 3274.4 milliseconds, RATIO(SERIAL/PARALLEL) = 0.252504 
// FOR N = 10000 AND MAX_VALUE = 100, TIME_BY_SERIAL_EXECUTION = 1201.8 milliseconds, TIME_BY_PARALLEL_EXECUTION = 1006.4 milliseconds, RATIO(SERIAL/PARALLEL) = 1.194157
// FOR N = 10000 AND MAX_VALUE = 1000, TIME_BY_SERIAL_EXECUTION = 1363.4 milliseconds, TIME_BY_PARALLEL_EXECUTION = 887.8 milliseconds, RATIO(SERIAL/PARALLEL) = 1.535706
// FOR N = 10000 AND MAX_VALUE = 10000, TIME_BY_SERIAL_EXECUTION = 3095.2 milliseconds, TIME_BY_PARALLEL_EXECUTION =  1192.2 milliseconds, RATIO(SERIAL/PARALLEL) = 2.596208

// FOR N = 100000 AND MAX_VALUE = 10, TIME_BY_SERIAL_EXECUTION = 10486 milliseconds, TIME_BY_PARALLEL_EXECUTION = 3253.2 milliseconds, RATIO(SERIAL/PARALLEL) = 3.223287 
// FOR N = 100000 AND MAX_VALUE = 100, TIME_BY_SERIAL_EXECUTION = 12767 milliseconds, TIME_BY_PARALLEL_EXECUTION = 4849.8 milliseconds, RATIO(SERIAL/PARALLEL) = 2.632479
// FOR N = 100000 AND MAX_VALUE = 1000, TIME_BY_SERIAL_EXECUTION = 17144 milliseconds, TIME_BY_PARALLEL_EXECUTION = 6081.4 milliseconds, RATIO(SERIAL/PARALLEL) = 2.819087
// FOR N = 100000 AND MAX_VALUE = 10000, TIME_BY_SERIAL_EXECUTION = 18899.6 milliseconds, TIME_BY_PARALLEL_EXECUTION =  6023.6 milliseconds, RATIO(SERIAL/PARALLEL) = 3.137592  










