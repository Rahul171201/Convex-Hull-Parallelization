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

double distance(pair<int,int> p1, pair<int,int> p2){
    double dis = sqrt((double(p2.second-p1.second)*double(p2.second-p1.second)) + (double(p2.first-p1.first)*double(p2.first-p1.first)));
    return dis;
}

vector<int> grahamsScan(vector<pair<int,int>> points){
    stack<pair<int,int>> s;
    vector<pair<int,int>> convex_hull;
    s.push(points[0]); convex_hull.push_back(points[0]);
    s.push(points[1]); convex_hull.push_back(points[1]);
    s.push(points[2]);

    for(int i=3;i<points.size();i++){
        
    }
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

bool compare(const pair<int, int> &a, const pair<int, int> &b)
{
    double angle1 = (double(a.second - bottom_most_point.second) / double(a.first - bottom_most_point.first));
    double angle2 = (double(b.second - bottom_most_point.second) / double(b.first - bottom_most_point.first));
    return (angle1 < angle2);
}

// Main function
int main()
{
    srand(time(0));
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
    points.push_back(start_point);

    for (int i = 0; i < P.n; i++)
    {
        if ((P.x[i] < start_point.first) || (P.x[i] == start_point.first))
            continue;
        else
            points.push_back({P.x[i], P.y[i]});
    }

    cout << "The bottom most point is = {" << start_point.first << ", " << start_point.second << "}\n";
    sort(points.begin(), points.end(), compare);

    for (int i = 0; i < P.n; i++)
    {
        if (P.x[i] == start_point.first && P.y[i] != start_point.second)
            points.push_back({P.x[i], P.y[i]});
    }

    vector<pair<int, int>> temp_clockwise_points;
    for (int i = 0; i < P.n; i++)
    {
        if ((P.x[i] > start_point.first) || (P.x[i] == start_point.first))
            continue;
        else
            temp_clockwise_points.push_back({P.x[i], P.y[i]});
    }
    sort(temp_clockwise_points.begin(), temp_clockwise_points.end(), compare);

    // pushing all elements on clockwise points vector to original points vector
    for (int i = 0; i < temp_clockwise_points.size(); i++)
        points.push_back(temp_clockwise_points[i]);

    // printing final points vector
    for (int i = 0; i < points.size(); i++)
        cout << points[i].first << " " << points[i].second << "\n";

    if(points.size() < 3){
        cout<<"It is not possible to have a convex hull with the given number of points\n";
        exit(0);
    }

    grahamsScan(points);
}