#include <bits/stdc++.h>
#include <pthread.h>

using namespace std;
using namespace std::chrono;

#define N 1024      // number of points in the plane
#define MAX_VAL 100 // maximum value of any point in the plane
#define no_of_threads 4

int x[N], y[N];
int part = 0;
int elements_per_thread = N / no_of_threads;
int p;
int q[no_of_threads];

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

void *threadFunction()
{
        // Each thread computes sum of 1/4th of array
    int thread_id = part++;
  
    for (int i = thread_id * (elements_per_thread); i < (thread_id + 1) * (elements_per_thread); i++){
        if (orientation(x[p], y[p], x[i], y[i], x[q[thread_id]], y[q[thread_id]]) == 2)
                q[thread_id] = i;
    }

    return NULL;
}

int main(int argc, char *argv[])
{

    srand(time(0));

    

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
    p = starting_point;
    int count = 0;
    do
    {
        // Add current point to result
        hull_x[count] = x[p];
        hull_y[count] = y[p];

        count++;

        // Search for a point 'q' such that orientation(p, q,
        // x) is counterclockwise for all points 'x'. The idea
        // is to keep track of last visited most counterclock-
        // wise point in q. If any point 'i' is more counterclock-
        // wise than q, then update q.

        for(int i=0;i<no_of_threads;i++)
            q[i] = (p + 1) % N;

        

        pthread_t threads[no_of_threads];

        // Creating threads
        for (int i = 0; i < no_of_threads; i++)
            pthread_create(&threads[i], NULL, threadFunction, (void *)NULL);

        // joining threads i.e. waiting for all threads to complete
        for (int i = 0; i < no_of_threads; i++)
            pthread_join(threads[i], NULL);

        int final_index = (p+1)%N; 
        for (int i = 0; i < no_of_threads; i++)
        {
            if (orientation(x[p], y[p], x[q[i]], y[q[i]], x[final_index], y[final_index]) == 2)
                final_index = q[i];
        }

        // Now q is the most counterclockwise with respect to p
        // Set p as q for next iteration, so that q is added to
        // result 'hull'
        p = final_index

    } while (p != starting_point); // While we don't come to first point
    // Get ending timepoint
    auto stop = high_resolution_clock::now();

    // Get duration. Substart timepoints to
    // get duration. To cast it to proper unit
    // use duration cast method
    auto duration = duration_cast<microseconds>(stop - start);

    cout << "Time taken by pThread Jarvis March Algorithm:  = "
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

    return 0;
}
