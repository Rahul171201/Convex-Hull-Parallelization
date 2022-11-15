#include <bits/stdc++.h>
#include "Plane.h"

using namespace std;

#define N 100        // number of points in the plane
#define MAX_VAL 1000 // maximum value of any point in the plane

pair<int, int> bottom_most_point = {0, 0};

// Function to find orientation of three points
// 0 --> p, q and r are collinear
// 1 --> Clockwise
// 2 --> Counterclockwise
int orientation(pair<int, int> p, pair<int, int> q, pair<int, int> r)
{
    int val = (q.second - p.second) * (r.first - q.first) - (q.first - p.first) * (r.second - q.second);

    if (val == 0)
        return 0;
    return (val > 0) ? 1 : 2;
}

// Function to get the starting point (bottom most point in the convex hull)
int getStartingPoint(vector<pair<int, int>> points)
{
    pair<int, int> point = {-1, -1};
    int min_y = INT_MAX, min_x = INT_MAX;
    int index = -1;
    for (int i = 0; i < points.size(); i++)
    {
        if (points[i].second < min_y)
        {
            min_y = points[i].second;
            min_x = points[i].first;
            index = i;
        }
        else if (points[i].second == min_y)
        {
            if (min_x > points[i].first)
                index = i;
            min_x = min(min_x, points[i].first);
        }
    }
    point = {min_x, min_y};
    return index;
}

// Function that defines the Jarvis March algorithm
vector<pair<int, int>> jarvisMarch(vector<pair<int, int>> points, int n)
{
    // There must be at least 3 points for a convex hull to be possible
    if (n < 3)
    {
        cout << "Convex Hull cannot be formed\n";
        exit(0);
    }

    vector<pair<int, int>> hull; // stores the convex hull points

    // Find the bottommost point
    int starting_point = getStartingPoint(points);

    int p = starting_point, q;
    do
    {
        // Add current point to the convex hull
        hull.push_back(points[p]);

        q = (p + 1) % n; // Let's say q is the most counter clockwise point

        for (int i = 0; i < n; i++)
        {
            // If i is more counterclockwise than current q, then update q
            if (orientation(points[p], points[i], points[q]) == 2)
                q = i;
        }

        p = q;

    } while (p != starting_point); // Repeat the process until we again reach the starting point

    return hull;
}

// Main function
int main()
{
    // srand(time(0));
    std::random_device rd;

    /* Random number generator */
    std::default_random_engine generator(rd());

    /* Distribution on which to apply the generator */
    std::uniform_int_distribution<long long unsigned> distribution(0, 0xFFFFFFFFFFFFFFFF);

    Plane P; // plane
    P.n = N;
    P.x = (int *)malloc(P.n * sizeof(int));
    P.y = (int *)malloc(P.n * sizeof(int));
    for (int i = 0; i < P.n; i++)
    {
        // int val_x = (rand() % (MAX_VAL - 1 + 1)) + 1;
        // int val_y = (rand() % (MAX_VAL - 1 + 1)) + 1;
        int val_x = (distribution(generator) % (MAX_VAL - 1 + 1)) + 1;
        int val_y = (distribution(generator) % (MAX_VAL - 1 + 1)) + 1;
        
        cout << val_x << " " << val_y << "\n";
        P.x[i] = val_x;
        P.y[i] = val_y;
    }

    cout << "The total number of points in the plane = " << P.n << "\n";

    // print the points in the plane
    // P.printPoints(P.x, P.y, P.n);

    vector<pair<int, int>> points;

    for (int i = 0; i < P.n; i++)
    {
        points.push_back({P.x[i], P.y[i]});
    }

    vector<pair<int, int>> convexHull = jarvisMarch(points, P.n);

    // cout << "The convex hull is created by the following points:\n";
    // for (auto it : convexHull)
    // {
    //     cout << "{" << it.first << ", " << it.second << "}\n";
    // }

    // Start writing the output onto a file
    ofstream MyFile("points.txt");

    MyFile << P.n << "\n";

    for (auto it : points)
    {
        MyFile << it.first << " " << it.second << "\n";
    }

    MyFile << convexHull.size() << "\n";

    for (auto it : convexHull)
    {
        MyFile << it.first << " " << it.second << "\n";
    }

    MyFile.close();

    // SYSTEM COMMAND TO RUN A PYTHON SCRIPT
    system("C:/Users/RAHUL/AppData/Local/Programs/Python/Python310/python.exe plot.py 1");
}