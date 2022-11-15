/*

Commands to Run:
    g++ graham_scan_sequential.cpp
    ./a.out
*/

#include<bits/stdc++.h>
#include<chrono>
#include<time.h>
#include<unistd.h>

using namespace std;
using namespace std::chrono;

#define br "\n"
#define int long long
#define pii pair<int,int>
#define piii pair<int,pii>
const int INF=1000000000;
const int NINF=-1000000000;

// UTILITY FUNCTIONS START
int get_random_int(int l,int r)
{
    mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());
    return uniform_int_distribution<int>(l,r)(rng);
}
void get_random_points(int *points,int n,int h)
{
    for(int i=0;i<2*n;i++)
        points[i]=get_random_int(1,h);
}
void print_array(vector<int>&arr)
{
    for(int i=0;i<arr.size();i++)
        cout<<arr[i]<<" ";
    cout<<br;
}
bool compare_array(int *arr1,int *arr2,int n)
{
    for(int i=0;i<n;i++)
    {
        if(arr1[i]!=arr2[i])
            return false;
    }
    return true;
}
bool compare_array_index(vector<int>&arr1,vector<int>&arr2,int *points)
{
    if(arr1.size()!=arr2.size())
        return false;
    for(int i=0;i<arr1.size();i++)
    {
        if(points[2*arr1[i]]!=points[2*arr2[i]]||points[2*arr1[i]+1]!=points[2*arr2[i]+1])
            return false;
    }
    return true;
}
void print_points(int *points,int n)
{
    for(int i=0;i<n;i++)
        cout<<points[2*i]<<" "<<points[2*i+1]<<br;
    cout<<br;
}
// UTILITY FUNCTIONS END

// Writes the points in points array in a file. 
// Used to write the random generated input points on which code is run
void write_input_file(int n,int *points,const char *filename)
{
    FILE *fout=fopen(filename,"w");
    fprintf(fout,"%lld\n",n);
    for(int i=0;i<n;i++)
        fprintf(fout,"%lld %lld\n",points[2*i],points[2*i+1]);
    fclose(fout);
}

// Writes the points in points array in a file. 
// Used to write the obtained convex hull points
void write_output_file(int n,vector<int>ind,int* points,const char *filename)
{
    FILE *fout=fopen(filename,"w");
    int ind_n=ind.size();
    fprintf(fout,"%lld\n",ind_n);
    for(int i=0;i<ind.size();i++)
        fprintf(fout,"%lld %lld\n",points[2*ind[i]],points[2*ind[i]+1]);
    fclose(fout);
}

// CHECK ORIENTATION FUNCTIONS START
// Checks if 3 points form a clockwise angle or not
int orientation(int ax,int ay,int bx,int by,int cx,int cy) 
{
    int v=ax*(by-cy)+bx*(cy-ay)+cx*(ay-by);
    if(v<0) return -1; // clockwise
    if(v>0) return +1; // counter-clockwise
    return 0;
}

//Checks if 3 points are collinear
bool collinear(int ax,int ay,int bx,int by,int cx,int cy) 
{ 
    return orientation(ax,ay,bx,by,cx,cy)==0; 
}

//Checks if 3 points are forming a clockwise angle
bool cw(int ax,int ay,int bx,int by,int cx,int cy)
{
    int o=orientation(ax,ay,bx,by,cx,cy);
    return o<=0;
}
// CHECK ORIENTATION FUNCTIONS END

vector<int> graham_scan_sequential(int *points,int n)
{
    vector<pair<int,pii>>v(n);
    for(int i=0;i<n;i++)
        v[i]={i,{points[2*i],points[2*i+1]}};
    piii mnv=*min_element(v.begin(),v.end(),[](piii a,piii b)
    {
        return make_pair(a.second.second,a.second.first)<make_pair(b.second.second,b.second.first);
    });
    sort(v.begin(),v.end(),[&mnv](const piii& a, const piii& b) 
    {
        int o=orientation(mnv.second.first,mnv.second.second,a.second.first,a.second.second,b.second.first,b.second.second);
        int temp_a1=(mnv.second.first-a.second.first);
        int temp_a2=(mnv.second.second-a.second.second);
        int temp_b1=(mnv.second.first-b.second.first);
        int temp_b2=(mnv.second.second-b.second.second);
        if(o==0)
            return temp_a1*temp_a1+temp_a2*temp_a2<temp_b1*temp_b1+temp_b2*temp_b2;
        return o<0;
    });
    int i=v.size()-1;
    while(i>=0&&collinear(mnv.second.first,mnv.second.second,v[i].second.first,v[i].second.second,v.back().second.first,v.back().second.second))
        i--;
    reverse(v.begin()+i+1,v.end());
    vector<int>st;
    for (int i=0;i<v.size();i++)
    {
        while(st.size()>1&&!cw(points[2*st[st.size()-2]],points[2*st[st.size()-2]+1],
                               points[2*st.back()],points[2*st.back()+1],
                               points[2*v[i].first],points[2*v[i].first+1]))
            st.pop_back();
        st.push_back(v[i].first);
    }
    return st;
}

int32_t main()
{
    // int max_num_blocks;
    // cout<<"Enter max number of blocks: ";
    // cin>>max_num_blocks;

    // FILE *fin=freopen("input.txt","r",stdin);
    
    int n;
    cout<<"Enter number of points: ";
    cin>>n;

    int h;
    cout<<"Enter max coordinate value: ";
    cin>>h;

    n=(1<<((int)floor(log2(n))));
    int *points=new int[2*n];
    get_random_points(points,n,h);

    write_input_file(n,points,"input_points.txt");

    cout<<"Number of points: "<<n<<br;
    cout<<"Max coordinate value: "<<h<<br;
    cout<<"Points:"<<br;
    print_points(points,n);

    auto start = high_resolution_clock::now();
    vector<int>res_seq=graham_scan_sequential(points,n);
    auto stop = high_resolution_clock::now();
    auto duration_sequential=duration_cast<microseconds>(stop-start);

    write_output_file(n,res_seq,points,"output_points.txt");

    cout<<br<<"***Sequential***"<<br;
    cout<<"Execution time: "<<duration_sequential.count()/1000.0<<" ms"<<br;
    print_array(res_seq);

    delete[] points;
}