#include<bits/stdc++.h>
 
using namespace std;

class Plane{
    public:
    int n; // number of points in the plane
    int *x; // set of x-coordinates
    int *y; // set of y-coordinates

    void printPoints(int *x, int *y, int n){
        cout<<"The set of points in the plane are:\n";
        for(int i=0;i<this->n;i++){
            cout<<"( "<<x[i]<<" , "<<y[i]<<" )\n";
        }
    }
};