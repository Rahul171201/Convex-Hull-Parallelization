/*

Commands to Run:
    nvcc graham_scan_parallel_cuda.cu
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
__host__ __device__ int orientation(int ax,int ay,int bx,int by,int cx,int cy) 
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

__global__ void kernel_function_copy(int *a,int *b,int n)
{
    int thread_id=blockIdx.x*blockDim.x+threadIdx.x;
    if(thread_id<n)
        b[thread_id]=a[thread_id];
}

// Return true if val(i)<=val(j), else false
__device__ int compare_index(int a,int b,int *points,int p0)
{
    if(a==b)
        return 0;
    if(a==NINF||b==INF)
        return -1;
    if(a==INF||b==NINF)
        return 1;
    if(points[2*a]==points[2*b]&&points[2*a+1]==points[2*b+1])
        return 0;
    int o=orientation(points[2*p0],points[2*p0+1],points[2*a],points[2*a+1],points[2*b],points[2*b+1]);
    int temp_a1=(points[2*p0]-points[2*a]);
    int temp_a2=(points[2*p0+1]-points[2*a+1]);
    int temp_b1=(points[2*p0]-points[2*b]);
    int temp_b2=(points[2*p0+1]-points[2*b+1]);
    if(o==0)
    {
        if(temp_a1*temp_a1+temp_a2*temp_a2<temp_b1*temp_b1+temp_b2*temp_b2)
            return -1;
        return 1;
    }
    if(o<0)
        return -1;
    return 1;
}
__device__ int get_min_index(int a,int b,int *points,int p0)
{
    if(compare_index(a,b,points,p0)<=0)
        return a;
    return b;
}
__device__ int get_max_index(int a,int b,int *points,int p0)
{
    if(compare_index(a,b,points,p0)<=0)
        return b;
    return a;
}

__device__ void find_merge_window(int *arr,int n,int *points,int p0,int start_a,int n_a,
                                  int start_b,int n_b,int num_elements,int *ret)
{
    int lo=start_a,hi=min(start_a+n_a,start_a+num_elements);
    // FFFFTTTT
    while(lo<hi)
    {
        int mid=(lo+hi)/2;
        int i=mid;
        int j=start_b+num_elements-(mid-start_a);
        if(j>start_b+n_b)
        {
            lo=mid+1;
            continue;
        }
        int last_a=((i!=start_a)?arr[i-1]:NINF);
        int last_b=((j!=start_b)?arr[j-1]:NINF);
        int cur_a=((i!=start_a+n_a)?arr[i]:INF);
        int cur_b=((j!=start_b+n_b)?arr[j]:INF);

        if(compare_index(get_min_index(cur_a,cur_b,points,p0),
                         get_max_index(last_a,last_b,points,p0),
                         points,p0)>=0)
        {
            ret[0]=i;
            ret[1]=j;
            return;
        }
        if(compare_index(cur_a,cur_b,points,p0)<=0)
            lo=mid+1;
        else
            hi=mid-1;
    }
    ret[0]=lo;
    ret[1]=start_b+num_elements-(lo-start_a);
}

__global__ void kernel_function_merge(int *arr,int *res,int n,int *points,int p0,int merge_size)
{
    int thread_id=blockIdx.x*blockDim.x+threadIdx.x;
    if(thread_id<n)
    {
        int start_a=(thread_id/merge_size)*merge_size;
        int n_a=merge_size/2;
        int start_b=(thread_id/merge_size)*merge_size+merge_size/2;
        int n_b=merge_size/2;

        int num_elements_cur=thread_id-start_a+1;
        int num_elements_old=num_elements_cur-1;
        int ret[2];
        find_merge_window(arr,n,points,p0,start_a,n_a,start_b,n_b,num_elements_old,ret);
        int l1=ret[0];
        int l2=ret[1];
        find_merge_window(arr,n,points,p0,start_a,n_a,start_b,n_b,num_elements_cur,ret);
        int r1=ret[0];
        int r2=ret[1];

        int i=l1,j=l2,k=start_a+num_elements_old;
        while(i!=r1&&j!=r2)
        {
            if(compare_index(arr[i],arr[j],points,p0)<=0)
                res[k++]=arr[i++];
            else
                res[k++]=arr[j++];
        }
        while(i!=r1)
            res[k++]=arr[i++];
        while(j!=r2)
            res[k++]=arr[j++];
    }
}

vector<int> graham_scan_cuda(int *points,int n,int num_blocks)
{
    vector<pair<int,pii>>v(n);
    for(int i=0;i<n;i++)
        v[i]={i,{points[2*i],points[2*i+1]}};
    piii mnv=*min_element(v.begin(),v.end(),[](piii a,piii b)
    {
        return make_pair(a.second.second,a.second.first)<make_pair(b.second.second,b.second.first);
    });
    int p0=mnv.first;

    int *res,*temp_res;
    cudaMallocManaged(&res,n*sizeof(int));
    cudaMallocManaged(&temp_res,n*sizeof(int));
    for(int i=0;i<n;i++)
        res[i]=i;
    for(int merge_size=2;merge_size<=n;merge_size*=2)
    {
        int num_threads=n;
        int num_threads_per_block=ceil((num_threads*1.0)/num_blocks);
        kernel_function_merge<<<num_blocks,num_threads_per_block>>>(res,temp_res,n,points,p0,merge_size);
        cudaDeviceSynchronize();
        kernel_function_copy<<<num_blocks,num_threads_per_block>>>(temp_res,res,n);
        cudaDeviceSynchronize();
    }
    cudaFree(res);
    cudaFree(temp_res);

    for(int i=0;i<n;i++)
        v[i]={res[i],{points[2*res[i]],points[2*res[i]+1]}};
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
    int max_num_blocks;
    cout<<"Enter max number of blocks: ";
    cin>>max_num_blocks;

    // FILE *fin=freopen("input.txt","r",stdin);
    
    int n;
    cout<<"Enter number of points: ";
    cin>>n;

    int h;
    cout<<"Enter max coordinate value: ";
    cin>>h;

    n=(1<<((int)floor(log2(n))));
    int *points;
    cudaMallocManaged(&points,(2*n)*sizeof(int));
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

    cout<<br<<"***Parallel CUDA***"<<br;
    bool best_flag=false;
    double best_time=10000000000;
    pii best_dimensions;
    for(int num_blocks=1;num_blocks<=max_num_blocks;num_blocks*=2)
    {
        int num_threads_per_block=ceil(1.0*n/num_blocks);
        if(num_threads_per_block>1024)
            continue;

        start = high_resolution_clock::now();
        vector<int>res_cuda=graham_scan_cuda(points,n,num_blocks);
        stop = high_resolution_clock::now();
        auto duration_parallel_cuda=duration_cast<microseconds>(stop-start);
        bool correct_flag=compare_array_index(res_seq,res_cuda,points);

        if(correct_flag&&duration_parallel_cuda.count()/1000.0<best_time)
        {
            best_time=duration_parallel_cuda.count()/1000.0;
            best_dimensions={num_blocks,num_threads_per_block};
            best_flag=true;
        }
    }
    cudaFree(points);

    cout<<"Best Execution Time: "<<best_time<<" ms"<<br;
    cout<<"Result: "<<(best_flag?"Correct!":"INCORRECT!!!")<<br;
    cout<<"Obtained for (num_blocks X num_threads_per_block) = ";
    cout<<"("<<best_dimensions.first<<" X "<<best_dimensions.second<<")"<<br<<br;
}