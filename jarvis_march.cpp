#include <bits/stdc++.h>
#include "Plane.h"

using namespace std;

#define N 10       // number of points in the plane
#define MAX_VAL 20 // maximum value of any point in the plane

pair<int, int> bottom_most_point = {0, 0};

// Function to know the orientation (clockwise or anticlockwise) of given 3 points
// returns 0 if collinear, return 1 if clockwise, return -1 if anticlockwise
int getCrossProduct(int a_x, int a_y, int b_x, int b_y, int c_x, int c_y)
{
    int val = ((b_y - a_y) * (c_x - b_x)) - ((b_x - a_x) * (c_y - a_y));
    if (val > 0)
        return 1;
    else if (val < 0)
        return -1;
    else
        return 0;
}

double distance(pair<int, int> p1, pair<int, int> p2)
{
    double dis = sqrt((double(p2.second - p1.second) * double(p2.second - p1.second)) + (double(p2.first - p1.first) * double(p2.first - p1.first)));
    return dis;
}

vector<pair<int, int>> jarvisMarch(vector<pair<int, int>> points)
{
    return points;
}

pair<int, int> getStartingPoint(int *x, int *y, int n)
{
    pair<int, int> point = {-1, -1};
    int min_y = INT_MAX, min_x = INT_MAX;
    for (int i = 0; i < n; i++)
    {
        if (y[i] < min_y)
        {
            min_y = y[i];
            min_x = x[i];
        }
        else if (y[i] == min_y)
        {
            min_x = min(min_x, x[i]);
        }
    }
    point = {min_x, min_y};
    return point;
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

    pair<int, int> start_point = getStartingPoint(P.x, P.y, P.n);

    bottom_most_point = start_point;

    vector<pair<int, int>> points;

    for (int i = 0; i < P.n; i++)
    {
        points.push_back({P.x[i], P.y[i]});
    }

    cout << "The bottom most point is = {" << start_point.first << ", " << start_point.second << "}\n";

    vector<pair<int, int>> convexHull = jarvisMarch(points);

    cout << "The convex hull is created by the following points:\n";
    for (auto it : convexHull)
    {
        cout << "{" << it.first << ", " << it.second << "}\n";
    }
}