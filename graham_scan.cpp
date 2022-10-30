#include <bits/stdc++.h>
#include "Plane.h"

using namespace std;

#define N 10       // number of points in the plane
#define MAX_VAL 20 // maximum value of any point in the plane

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

}