#include <bits/stdc++.h>
#include "Plane.h"

using namespace std;

#define N 10       // number of points in the plane
#define MAX_VAL 20 // maximum value of any point in the plane

pair<int, int> bottom_most_point = {0, 0};

// To find orientation of ordered triplet (p, q, r).
// The function returns following values
// 0 --> p, q and r are collinear
// 1 --> Clockwise
// 2 --> Counterclockwise
int orientation(pair<int, int> p, pair<int, int> q, pair<int, int> r)
{
    int val = (q.second - p.second) * (r.first - q.first) - (q.first - p.first) * (r.second - q.second);

    if (val == 0)
        return 0;             // collinear
    return (val > 0) ? 1 : 2; // clock or counterclock wise
}

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

// Prints convex hull of a set of n points.
vector<pair<int, int>> jarvisMarch(vector<pair<int, int>> points, int n)
{
    // There must be at least 3 points
    if (n < 3)
    {
        cout << "Jarvis March is not possible\n";
        exit(0);
    }

    // Initialize Result
    vector<pair<int, int>> hull;

    // Find the leftmost point
    int starting_point = getStartingPoint(points);

    // Start from leftmost point, keep moving counterclockwise
    // until reach the start point again.  This loop runs O(h)
    // times where h is number of points in result or output.
    int p = starting_point, q;
    do
    {
        // Add current point to result
        hull.push_back(points[p]);

        // Search for a point 'q' such that orientation(p, q,
        // x) is counterclockwise for all points 'x'. The idea
        // is to keep track of last visited most counterclock-
        // wise point in q. If any point 'i' is more counterclock-
        // wise than q, then update q.
        q = (p + 1) % n;
        for (int i = 0; i < n; i++)
        {
            // If i is more counterclockwise than current q, then
            // update q
            if (orientation(points[p], points[i], points[q]) == 2)
                q = i;
        }

        // Now q is the most counterclockwise with respect to p
        // Set p as q for next iteration, so that q is added to
        // result 'hull'
        p = q;

    } while (p != starting_point); // While we don't come to first point

    return hull;
}

// Main function
int main()
{
    // srand(time(0));
    Plane P;
    P.n = N;
    P.x = (int *)malloc(P.n * sizeof(int));
    P.y = (int *)malloc(P.n * sizeof(int));
    for (int i = 0; i < P.n; i++)
    {
        int val_x = (rand() % (MAX_VAL - 1 + 1)) + 1;
        int val_y = (rand() % (MAX_VAL - 1 + 1)) + 1;
        P.x[i] = val_x;
        P.y[i] = val_y;
    }

    // total number of points
    cout << "The total number of points in the plane = " << P.n << "\n";

    // print the points in the plane
    P.printPoints(P.x, P.y, P.n);

    vector<pair<int, int>> points;

    for (int i = 0; i < P.n; i++)
    {
        points.push_back({P.x[i], P.y[i]});
    }

    // cout << "The bottom most point is = {" << start_point.first << ", " << start_point.second << "}\n";

    vector<pair<int, int>> convexHull = jarvisMarch(points, P.n);

    cout << "The convex hull is created by the following points:\n";
    for (auto it : convexHull)
    {
        cout << "{" << it.first << ", " << it.second << "}\n";
    }

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
}