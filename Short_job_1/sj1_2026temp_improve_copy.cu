#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>



static void HandleError( cudaError_t err, const char *file, int line ) {

  if (err != cudaSuccess) {
    printf( "%s in %s at line %d\n", cudaGetErrorString( err ), file, line );
      exit( EXIT_FAILURE );
  }
}

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))


int find(char *s1,char *s2)
{
  int i,l1,l2;
  l1=strlen(s1);
  l2=strlen(s2);
  for(i=0;i<=(l1-l2);i++)
    if (strncmp(s1+i,s2,l2)==0) return i;
  return -1;  
}



// zacatek casti k modifikaci
// beginning of part for modification

// muzete pridat vlastni funkce nebo datove struktury,
// you can also add new functions or data structures
__global__ void fused_kernel(float* __restrict__ A,
                             float* __restrict__ B,
                             float* __restrict__ C,
                             int N)
{
    __shared__ float sA[32][32];
    __shared__ float sB[32];

    int tx = threadIdx.x;
    int bx = blockIdx.x;

    int lane = tx % 32;
    int warp_id = tx / 32;

    int warps_per_block = blockDim.x / 32;

    for (int offset = N - 32; offset >= 0; offset -= 32)
    {
        // =========================
        // 1. LOAD B (einmal pro Block)
        // =========================
        if (tx < 32)
            sB[tx] = B[offset + tx];

        __syncthreads();

        // =========================
        // 2. TRIANGULAR SOLVE (nur Block 0)
        // =========================
        if (bx == 0)
        {
            // load A block
            if (tx < 32)
            {
                for (int i = 0; i < 32; i++)
                    sA[i][tx] = A[(offset + i)*N + (offset + tx)];
            }

            __syncthreads();

            // backward substitution
            for (int i = 31; i >= 0; i--)
            {
                float sum = 0.0f;

                // parallel reduction innerhalb Warp
                if (warp_id == 0)
                {
                    for (int j = i+1 + lane; j < 32; j += 32)
                        sum += sA[i][j] * sB[j];

                    for (int d = 16; d > 0; d /= 2)
                        sum += __shfl_down_sync(0xffffffff, sum, d);

                    if (lane == 0)
                        sB[i] = (C[offset+i] - sum) / sA[i][i];
                }

                __syncthreads();
            }

            // store result
            if (tx < 32)
                B[offset + tx] = sB[tx];
        }

        __syncthreads(); // warten bis tri fertig

        // =========================
        // 3. RECT UPDATE (ALLE BLOECKE)
        // =========================

        int row = bx * warps_per_block + warp_id;

        if (row < offset)
        {
            float sum = 0.0f;

            const float* Arow = &A[row * N + offset];

            // vectorized loads
            const float4* A4 = (const float4*)Arow;
            const float4* B4 = (const float4*)sB;

            // alle Threads arbeiten!
            for (int j = lane; j < 8; j += 32)
            {
                float4 a = A4[j];
                float4 b = B4[j];

                sum += a.x * b.x;
                sum += a.y * b.y;
                sum += a.z * b.z;
                sum += a.w * b.w;
            }

            // warp reduction
            for (int d = 16; d > 0; d /= 2)
                sum += __shfl_down_sync(0xffffffff, sum, d);

            if (lane == 0)
                C[row] -= sum;
        }

        __syncthreads(); // nächste Iteration
    }
}

  /*
  vylepsete vykonnost tohoto volani
  improve performance of this call
*/

void problem(float* devA, float* devB, float* devC, int N)
{
    int THREADS = 512;
    int warps_per_block = THREADS / 32;
    int BLOCKS = (N + warps_per_block - 1) / warps_per_block;

    fused_kernel<<<BLOCKS, THREADS>>>(devA, devB, devC, N);
}


// end of part for modification
// konec casti k modifikaci





int main( void ) {
 int N,i,j;
 cudaEvent_t start, stop;
 float elapsedTime;

 //cudaStream_t stream;
    float *hostA,*hostB,*hostB2,*hostC;
    float *devA,*devB,*devC;
    int diag[]={-4,-2,-1,1,2,4};
 double tt,limit; 
 int res,cfg=0;
  char *s;
  char sub1[] = "4070";
  char sub2[] = "A100";

 srand(time(NULL));

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device name: %s\n", prop.name);
        printf("CC: %d.%d\n",  prop.major,prop.minor);  
    printf("#SM: %d\n",  prop.multiProcessorCount);           
  s=prop.name;
  limit=100.0;
  res = find(s,sub1);
  
  if (res >=0)
      {	cfg=1; limit=12.3; }
  res = find(s,sub2);      
  if (res >=0)
      {	cfg=2; limit=23.8; }

 
  // start the timers
  HANDLE_ERROR( cudaEventCreate( &start ) );
  HANDLE_ERROR( cudaEventCreate( &stop ) );

  int bar;
  int err=0;
  tt=0.0;
  for(int mea=0;mea<3;mea++)     
  {
    N=(mea*11+3)*1024;   
    HANDLE_ERROR( cudaMalloc( (void**)&devA, N * N* sizeof(float) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&devB, N * sizeof(float) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&devC, N * sizeof(float) ) );

    hostA=(float *)malloc(N * N* sizeof(float));
    HANDLE_ERROR( cudaHostAlloc( (void**)&hostB, N * sizeof(float), cudaHostAllocDefault ) );
    HANDLE_ERROR( cudaHostAlloc( (void**)&hostC, N * sizeof(float), cudaHostAllocDefault ) );

    HANDLE_ERROR( cudaHostAlloc( (void**)&hostB2, N * sizeof(float), cudaHostAllocDefault ) );
    for (i=0; i<N; i++)
      hostB[i]=rand()%19-9;

    for (i=0; i<N; i++)
    {
        //hostX[i] = ((rand()%1024)-512)/64.0;
        //hostY[i] = ((rand()%1024)-512)/64.0;
      float x=0;
      for (j=0; j<N; j++)
      {
        if (j<i) hostA[i*N+j]=0.0;
        if (i==j) hostA[i*N+j]=diag[rand()%6];
        if (j>i) hostA[i*N+j]=rand()%7-3;
        x+=hostA[i*N+j]*hostB[j];
      }
      hostC[i]=x;
    }    
      
      /*
    auto end2 = std::chrono::steady_clock::now();    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end2 - start2);

    std::cout << "CPU Time elapsed: " << duration.count() << " us" << std::endl;
    fflush(stdout);
    **/

    cudaMemcpy(devA,hostA,sizeof(float)*N *N,cudaMemcpyHostToDevice);
    cudaMemcpy(devC,hostC,sizeof(float)*N,cudaMemcpyHostToDevice);

    HANDLE_ERROR(cudaMemset(devB, 0, N*sizeof(float)));
    HANDLE_ERROR(cudaDeviceSynchronize());

    HANDLE_ERROR( cudaEventRecord( start, 0 ) );

    problem(devA, devB, devC, N);

    HANDLE_ERROR(cudaDeviceSynchronize());
    HANDLE_ERROR( cudaEventRecord( stop, 0 ) );
    HANDLE_ERROR( cudaEventSynchronize( stop ) );
    HANDLE_ERROR( cudaEventElapsedTime( &elapsedTime, start, stop ) );
    printf( "%i Time taken:  %g ms\n", N,elapsedTime );
    fflush(stdout);
   tt+=elapsedTime;

    cudaMemcpy(hostB2,devB,sizeof(float)*N,cudaMemcpyDeviceToHost);
    j=0;
    for (i=0; i<N; i++)
    {
      if (fabs(hostB[i]-hostB2[i])>1e-1)  j++;
      //printf("%d %5.3g %5.3g\n",i,hostB[i],hostB2[i]);
    }
    printf("%d errors=%d\n",N,j);
    //printf("\n");


    // cleanup the streams and memory

    free( hostA );
    HANDLE_ERROR( cudaFreeHost( hostB ) );
    HANDLE_ERROR( cudaFreeHost( hostC ) );


    HANDLE_ERROR( cudaFree( devA ) );
    HANDLE_ERROR( cudaFree( devB ) );
    HANDLE_ERROR( cudaFree( devC ) );

  if ((j>1)||(tt>(5.0*limit)))
  {
    printf("Error(s) or too slow => NO points\n");
  HANDLE_ERROR( cudaEventDestroy( start ) );
  HANDLE_ERROR( cudaEventDestroy( stop ) );

    return 0;

  }   

  } // mea-for
  if ((cfg==1)||(cfg==2))
  {
  float tmp;
  tmp=12.0*limit/tt;
  printf("Total GPU time %g ms\n",tt);
  if (tmp>15.0) tmp=15.0;
  printf("SJ1 %g points\n",tmp);
  }
  else
  {
    printf("Total GPU time %g ms\n",tt);
    printf("Unsupported GPU\n");
  }
  
  HANDLE_ERROR( cudaEventDestroy( start ) );
  HANDLE_ERROR( cudaEventDestroy( stop ) );

  // cleanup the streams and memory


  return 0;
}
