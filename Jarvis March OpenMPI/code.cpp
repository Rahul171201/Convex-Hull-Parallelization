#include <bits/stdc++.h>
#include <mpi.h>

using namespace std;
using namespace std::chrono;

#define N 1024      // number of points in the plane
#define MAX_VAL 100 // maximum value of any point in the plane

// To find orientation of ordered triplet (p, q, r).
// The function returns following values
// 0 --> p, q and r are collinear
// 1 --> Clockwise
// 2 --> Counterclockwise

int orientation(int p_x, int p_y, int q_x, int q_y, int r_x, int r_y)
{
    int val = (q_y - p_y) * (r_x - q_x) - (q_x - p_x) * (r_y - q_y);

    if (val == 0)
        return 0;             // collinear
    return (val > 0) ? 1 : 2; // clock or counterclock wise
}

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

int main(int argc, char *argv[])
{

    int pid, np, elements_per_process, n_elements_recieved;
    // np -> no. of processes
    // pid -> process id

    MPI_Status status;

    // Creation of parallel processes
    MPI_Init(&argc, &argv);

    // find out process ID,
    // and how many processes were started
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    MPI_Comm_size(MPI_COMM_WORLD, &np);

    // master process
    if (pid == 0)
    {

        srand(time(0));

        int x[N], y[N];

        for (int i = 0; i < N; i++)
        {
            int val_x = (rand() % (MAX_VAL - 1 + 1)) + 1;
            int val_y = (rand() % (MAX_VAL - 1 + 1)) + 1;
            x[i] = val_x;
            y[i] = val_y;
        }

        // total number of points
        cout << "The total number of points in the plane = " << N << "\n";

        cout << "The points are\n";
        for (int i = 0; i < N; i++)
        {
            cout << x[i] << " " << y[i] << "\n";
        }

        if (np > 1)
        {
            // There must be at least 3 points
            if (N < 3)
            {
                cout << "Jarvis March is not possible\n";
                exit(0);
            }

            // Get starting timepoint
            auto start = high_resolution_clock::now();

            int hull_x[N];
            int hull_y[N];

            // Find the leftmost point
            int starting_point = getStartingPoint(x, y, N);

            // Start from leftmost point, keep moving counterclockwise
            // until reach the start point again.  This loop runs O(h)
            // times where h is number of points in result or output.
            int p = starting_point, q;
            int count = 0;

            int again = 1;

            do
            {
                for (int i = 1; i < np; i++)
                {
                    MPI_Send(&again, 1, MPI_INT, i, i - 1, MPI_COMM_WORLD);
                }

                // Add current point to result
                hull_x[count] = x[p];
                hull_y[count] = y[p];

                count++;

                // Search for a point 'q' such that orientation(p, q,
                // x) is counterclockwise for all points 'x'. The idea
                // is to keep track of last visited most counterclock-
                // wise point in q. If any point 'i' is more counterclock-
                // wise than q, then update q.

                int elements_per_process = (N / np);
                q = (p + 1) % N;

                // cout << "Message sending started for loop number =  " << count << " with value of p = "<<p<<"\n";

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
            auto stop = high_resolution_clock::now();

            // Get duration. Substart timepoints to
            // get duration. To cast it to proper unit
            // use duration cast method
            auto duration = duration_cast<microseconds>(stop - start);

            cout << "Time taken by OpenMPI Jarvis March Algorithm:  = "
                 << duration.count() << " microseconds" << endl;

            int convex_hull[2 * count];
            for (int i = 0; i < count; i = i + 1)
            {
                convex_hull[2 * i] = hull_x[i];
                convex_hull[2 * i + 1] = hull_y[i];
            }

            cout << "The points in the convex hull are\n";
            for (int i = 0; i < 2 * count; i = i + 2)
            {
                cout << convex_hull[i] << " " << convex_hull[i + 1] << "\n";
            }

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
