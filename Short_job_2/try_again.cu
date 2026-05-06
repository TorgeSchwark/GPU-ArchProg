#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cooperative_groups.h>

#define BLOCK_SIZE 32
constexpr int baseN = 16;

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


void CPU( int *inA, char *inB, int N ) {
  // in-place exclusive segmented sum
  int i,s;

  s=0;
  for(i=0;i<N;i++)
  {
    s+=inA[i];
    if (inB[i+1]) {
      inA[i]=s;
      s=0.0; 
    }
    else
      inA[i]=0;
  }    
}

// zacatek casti k modifikaci
// beginning of part for modification

// muzete pridat vlastni funkce nebo datove struktury,
// you can also add new functions or data structures

// Note:
//  you can use tmpA, tmpB arrays for your purposes

__device__ void segmented_scan_block(
    int *a,
    int *tmpa,
    char* tmpb,
    char *b,
    int n
) {
  int tid = threadIdx.x;

  tmpa[tid] = a[tid];
  tmpb[tid] = b[tid];

  // 👉 Segment-Ende markieren (B[tid+1]!)
  
  for (int offset = 1; offset < n; offset <<= 1) {

    __syncthreads();

    if (tid >= offset) {
      if (!tmpb[tid]) {
        tmpa[tid] = a[tid] + a[tid - offset];

        if (tmpb[tid - offset]) {
          tmpb[tid] = tmpb[tid - offset];
        }
      }
    }

    __syncthreads();

    a[tid] = tmpa[tid];
  }

  __syncthreads();
}



__device__ void segmented_scan_block2(
    int *a,
    int *tmpa,
    int* tmpb,
    int *b,
    int n
) {
  int tid = threadIdx.x;

  tmpa[tid] = a[tid];
  tmpb[tid] = b[tid];

  // 👉 Segment-Ende markieren (B[tid+1]!)
  
  for (int offset = 1; offset < n; offset <<= 1) {

    __syncthreads();

    if (tid >= offset) {
      if (!tmpb[tid]) {
        tmpa[tid] = a[tid] + a[tid - offset];

        if (tmpb[tid - offset]) {
          tmpb[tid] = tmpb[tid - offset];
        }
      }
    }

    __syncthreads();

    a[tid] = tmpa[tid];
  }

  __syncthreads();
}


__global__ void kernel1(int* A, char* B, int N, int* blockSum, int* blockFlag){

  extern __shared__ unsigned char shared[];

  __shared__ int first_end_pos;

  int* sA = (int*) shared;
  int* tmp = (int*)&sA[blockDim.x];
  char* sB = (char*)&tmp[blockDim.x];
  char* tmpb = (char*)&sB[blockDim.x];

  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  int tid = threadIdx.x;

  if (tid == 0) first_end_pos = INT_MAX;
  __syncthreads();

  // load
  if (gid < N) {
    sA[tid] = A[gid];
    sB[tid] = B[gid];
  } else {
    sA[tid] = 0;
    sB[tid] = 1;
  }

  __syncthreads();

  // scan
  segmented_scan_block(sA, tmp, tmpb, sB, blockDim.x);

  __syncthreads();

  // write back
  if (gid < N) {
    if (B[gid + 1]) {  // verhindert, das wir einen Überhang haben, der in den eigenen Block gehört!
      A[gid] = sA[tid];
    } else {
      A[gid] = 0;
    }
  }

   if (gid < N && B[gid+1]){
      atomicMin(&first_end_pos ,tid);
  }
  __syncthreads();

  // 👉 Ergebnis speichern
  if (tid == blockDim.x - 1) {

    blockFlag[blockIdx.x] = first_end_pos;      // -1 wenn kein Bit gesetzt

    if (gid < N && !B[gid + 1]) { // verhindert, das wir einen Überhang haben, der in den eigenen Block gehört!
      blockSum[blockIdx.x] = sA[tid];
    } else {
      blockSum[blockIdx.x] = 0;
    }
  }
}

__global__ void kernel2(int* A, char* B, int N ,int* blockSum, int* blockFlag, int lvl, int globaln){

  extern __shared__ int sharedd[];
  __shared__ int first_end_pos;

  int* useA = (int*)sharedd;
  int* tmpA = (int*)&useA[blockDim.x];
  int* useB = (int*)&tmpA[blockDim.x];
  int* tempB = (int*)&useB[blockDim.x];

  // 🔒 shared copy von blockFlag (wichtig!)
  __shared__ int sFlag[BLOCK_SIZE + 1];

  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  int tid = threadIdx.x;

  if (tid == 0) first_end_pos = INT_MAX;
  __syncthreads();

  // Load blockSum / blockFlag
  if (gid < N) {
    useA[tid] = blockSum[gid];
    useB[tid] = (blockFlag[gid] != INT_MAX);
    sFlag[tid] = blockFlag[gid];
  } else {
    useA[tid] = 0;
    useB[tid] = 1;
    sFlag[tid] = INT_MAX;
  }

  // wichtig für gid+1 Zugriff
  if (tid == blockDim.x - 1) {
    if (gid + 1 < N)
      sFlag[tid + 1] = blockFlag[gid + 1];
    else
      sFlag[tid + 1] = INT_MAX;
  }

  __syncthreads();

  // stride berechnen
  int stride = 1;
  for (int i = 0; i < lvl; i++) stride *= blockDim.x;

  // =========================
  // scan
  // =========================
  segmented_scan_block2(useA, tmpA, tempB, useB, blockDim.x);

  __syncthreads();

  // =========================
  // 🔥 FIXED WRITE (atomic + shared flag)
  // =========================
  if (tid + 1 < blockDim.x && gid < N - 1) {

    int nextFlag = sFlag[tid + 1];

    if (nextFlag != INT_MAX) {

      int index = (gid + 1) * stride + nextFlag;

      if (index < globaln) {
        atomicAdd(&A[index], useA[tid]);   // 🔥 entscheidend!
      }
    }
  }

  // =========================
  // next level vorbereiten
  // =========================
  int val = INT_MAX;

  if (gid < N && sFlag[tid] != INT_MAX) {
    val = sFlag[tid] + tid * stride;
  }

  atomicMin(&first_end_pos, val);

  __syncthreads();

  if (tid == blockDim.x - 1) {

    blockFlag[blockIdx.x] = first_end_pos;

    if (gid < N) {
      blockSum[blockIdx.x] = useA[tid];
    }
  }
}

// // lvl = 1 first call kernel2 block_dim muss 2^n sein!!
// __global__ void kernel2(int* A, char* B, int N ,int* blockSum, int* blockFlag, int lvl, int globaln){

//   extern __shared__ int sharedd[];
//   __shared__ int first_end_pos;

//   int* useA = (int*)sharedd;
//   int* tmpA = (int*)&useA[blockDim.x];
//   int* useB = (int*)&tmpA[blockDim.x];
//   int* tempB = (int*)&useB[blockDim.x];

//   int gid = blockIdx.x * blockDim.x + threadIdx.x;
//   int tid = threadIdx.x;

//   if (tid == 0) first_end_pos = INT_MAX;
//   __syncthreads();

//   if (gid < N) {
//     useA[tid] = blockSum[gid];
//     useB[tid] = blockFlag[gid] != INT_MAX;
//   } else {
//     useA[tid] = 0;
//     useB[tid] = 1; 
//   }
//   int stride = 1;
//   for (int i = 0; i < lvl; i++) stride *= blockDim.x;

  

//   __syncthreads();

//   // scan
//   segmented_scan_block2(useA, tmpA, tempB, useB, blockDim.x);

//   __syncthreads();

//   if(tid+1 < blockDim.x && gid < N-1 && blockFlag[gid+1] != INT_MAX && ((gid+1) * stride + blockFlag[gid+1]) < globaln){
//     A[(gid+1) * stride + blockFlag[gid+1]] += useA[tid];
//   }

  

//   // if we have overhead we can add in the block we do
  

//   int val = INT_MAX;
//   if (gid < N && blockFlag[gid] != INT_MAX) {
//     val = blockFlag[gid] + tid * stride;
//   }

//   atomicMin(&first_end_pos, val);
//   __syncthreads();

//   if (tid == blockDim.x - 1) {

//     blockFlag[blockIdx.x] = first_end_pos;      // -1 wenn kein Bit gesetzt

//     if (gid < N) {
      
//       blockSum[blockIdx.x] = useA[tid];
      
//     }
//   }
  
// }


__global__ void ess(int* inA, char *inB, int N, int *tmpA, char *tmpB)
{

  int i,s;

  s=0;
  for(i=0;i<N;i++)
  {
    s+=inA[i];
    if (inB[i+1]) {
      inA[i]=s;
      s=0.0; 
    }
    else
      inA[i]=0;
  }    
}

void ess_Gpu(int* inA, char *inB, int N, int *tmpA, char *tmpB)
{
    int blockSize = BLOCK_SIZE;
    int numBlocks = (N + blockSize - 1) / blockSize;

    int *blockSum;
    int *blockFlag;

    cudaMalloc(&blockSum, numBlocks * sizeof(int));
    cudaMalloc(&blockFlag, numBlocks * sizeof(int));

    size_t sharedSize = blockSize * (sizeof(int)*2 + sizeof(char)*2);

    // =========================
    // KERNEL 1
    // =========================
    kernel1<<<numBlocks, blockSize, sharedSize>>>(inA, inB, N, blockSum, blockFlag);
    cudaDeviceSynchronize();

    // =========================
    // KERNEL 2 (hierarchisch)
    // =========================
    int lvl = 1;
    int stride = blockSize;
    sharedSize = blockSize * (sizeof(int)*4);

    while (stride < N)
    {
        int n = (N + stride - 1) / stride;
        int numBlocks2 = (n + blockSize - 1) / blockSize;

        kernel2<<<numBlocks2, blockSize, sharedSize>>>(
            inA, inB, n, blockSum, blockFlag, lvl, N
        );
        cudaDeviceSynchronize();

        lvl++;
        stride *= blockSize;
    }

    cudaFree(blockSum);
    cudaFree(blockFlag);
}

void ess_Gpu_test(int* inA, char *inB, int N, int *tmpA, char *tmpB)
{
    int blockSize = BLOCK_SIZE;
    int numBlocks = (N + blockSize - 1) / blockSize;

    int *blockSum;
    int *blockFlag;

    cudaMalloc(&blockSum, numBlocks * sizeof(int));
    cudaMalloc(&blockFlag, numBlocks * sizeof(int));

    size_t sharedSize = blockSize * (sizeof(int)*2 + sizeof(char)*2);

    // Debug buffers (max Größe)
    int* h_A = (int*)malloc(N * sizeof(int));
    char* h_B = (char*)malloc((N+1) * sizeof(char));
    int* h_blockSum = (int*)malloc(numBlocks * sizeof(int));
    int* h_blockFlag = (int*)malloc(numBlocks * sizeof(int));

    // =========================
    // KERNEL 1
    // =========================
    kernel1<<<numBlocks, blockSize, sharedSize>>>(inA, inB, N, blockSum, blockFlag);
    cudaDeviceSynchronize();

    printf("\n===== After kernel1 =====\n");

    cudaMemcpy(h_A, inA, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_B, inB, (N+1) * sizeof(char), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_blockSum, blockSum, numBlocks * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_blockFlag, blockFlag, numBlocks * sizeof(int), cudaMemcpyDeviceToHost);

    printf("A: ");
    for(int i = 0; i < N; i++) printf("%d ", h_A[i]);
    printf("\n");

    printf("B: ");
    for(int i = 0; i < N+1; i++) printf("%d ", h_B[i]);
    printf("\n");

    printf("blockSum: ");
    for(int i = 0; i < numBlocks; i++) printf("%d ", h_blockSum[i]);
    printf("\n");

    printf("blockFlag: ");
    for(int i = 0; i < numBlocks; i++) printf("%d ", h_blockFlag[i]);
    printf("\n");

    // =========================
    // KERNEL 2 (hierarchisch)
    // =========================
    int lvl = 1;
    int stride = blockSize;
    sharedSize = blockSize * (sizeof(int)*4);

    while (stride < N)
    {
        int n = (N + stride - 1) / stride;
        int numBlocks2 = (n + blockSize - 1) / blockSize;

        // neue Host-Buffer für diese Ebene
        int* h_blockSum2 = (int*)malloc(numBlocks2 * sizeof(int));
        int* h_blockFlag2 = (int*)malloc(numBlocks2 * sizeof(int));

        kernel2<<<numBlocks2, blockSize, sharedSize>>>(
            inA, inB, n, blockSum, blockFlag, lvl, N
        );
        cudaDeviceSynchronize();

        printf("\n===== After kernel2 (lvl=%d, stride=%d) =====\n", lvl, stride);

        cudaMemcpy(h_A, inA, N * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_B, inB, (N+1) * sizeof(char), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_blockSum2, blockSum, numBlocks2 * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_blockFlag2, blockFlag, numBlocks2 * sizeof(int), cudaMemcpyDeviceToHost);

        printf("A: ");
        for(int i = 0; i < N; i++) printf("%d ", h_A[i]);
        printf("\n");

        printf("B: ");
        for(int i = 0; i < N+1; i++) printf("%d ", h_B[i]);
        printf("\n");

        printf("blockSum: ");
        for(int i = 0; i < numBlocks2; i++) printf("%d ", h_blockSum2[i]);
        printf("\n");

        printf("blockFlag: ");
        for(int i = 0; i < numBlocks2; i++) printf("%d ", h_blockFlag2[i]);
        printf("\n");

        free(h_blockSum2);
        free(h_blockFlag2);

        lvl++;
        stride *= blockSize;
    }

    // cleanup
    free(h_A);
    free(h_B);
    free(h_blockSum);
    free(h_blockFlag);

    cudaFree(blockSum);
    cudaFree(blockFlag);
}

#include <stdio.h>
#include <cuda_runtime.h>


void printArrayInt(const char* name, int* arr, int n) {
    printf("%s: ", name);
    for (int i = 0; i < n; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");
}

void printArrayChar(const char* name, char* arr, int n) {
    printf("%s: ", name);
    for (int i = 0; i < n; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");
}
void test_repeat(int repeat) {
    const int baseN = 16;
    const int N = baseN * repeat;

    int *h_inA  = (int*)malloc(N * sizeof(int));
    char *h_inB = (char*)malloc((N+1) * sizeof(char));

    // Basisdaten
    int baseA[baseN] = {
        0,1,2,3,4,5,6,7,
        8,9,10,11,12,13,14,15
    };

    // Wichtig: Segment geht über Blockgrenzen!
    char baseB[baseN+1] = {
        1,0,0,0, 0,0,0,0,
        0,0,0,0, 0,0,0,1,  // Segment endet hier
        1
    };

    // Pattern wiederholen
    for (int r = 0; r < repeat; r++) {
        for (int i = 0; i < baseN; i++) {
            h_inA[r * baseN + i] = baseA[i];
            h_inB[r * baseN + i] = baseB[i];
        }
    }

    // Letztes Element MUSS 1 sein
    h_inB[N] = 1;

    // =========================
    // CPU Referenz
    // =========================
    int *h_ref = (int*)malloc(N * sizeof(int));
    memcpy(h_ref, h_inA, N * sizeof(int));

    CPU(h_ref, h_inB, N);

    // =========================
    // GPU Setup
    // =========================
    int *d_inA;
    char *d_inB;

    cudaMalloc(&d_inA, N * sizeof(int));
    cudaMalloc(&d_inB, (N+1) * sizeof(char));

    cudaMemcpy(d_inA, h_inA, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_inB, h_inB, (N+1) * sizeof(char), cudaMemcpyHostToDevice);

    int *tmpA;
    char *tmpB;
    cudaMalloc(&tmpA, N * sizeof(int));
    cudaMalloc(&tmpB, N * sizeof(char));

    // =========================
    // GPU Run
    // =========================
    ess_Gpu(d_inA, d_inB, N, tmpA, tmpB);

    // =========================
    // Ergebnis zurückholen
    // =========================
    int *h_out = (int*)malloc(N * sizeof(int));
    cudaMemcpy(h_out, d_inA, N * sizeof(int), cudaMemcpyDeviceToHost);

    // =========================
    // Fehleranalyse
    // =========================
    int errors = 0;

    for (int i = 0; i < N; i++) {
        if (h_out[i] != h_ref[i]) {
            if (errors < 20) {
                printf("Error at %d: GPU=%d CPU=%d\n",
                       i, h_out[i], h_ref[i]);
            }
            errors++;
        }
    }

    printf("Total errors: %d / %d\n", errors, N);

    // Kleine Ausgabe zum Debuggen
    if (N <= 64) {
        printf("\nCPU:\n");
        for (int i = 0; i < N; i++) printf("%d ", h_ref[i]);
        printf("\nGPU:\n");
        for (int i = 0; i < N; i++) printf("%d ", h_out[i]);
        printf("\n");
    }

    // =========================
    // Cleanup
    // =========================
    free(h_inA);
    free(h_inB);
    free(h_ref);
    free(h_out);

    cudaFree(d_inA);
    cudaFree(d_inB);
    cudaFree(tmpA);
    cudaFree(tmpB);
}

// void test() {
//     int N = 16;

//     int h_inA[N]  = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
//     char h_inB[N+1] = {1,1,0,0, 0,0,1,0, 1,0,0,1, 0,0,1,0, 1};

//     int *d_inA;
//     char *d_inB;

//     cudaMalloc(&d_inA, N * sizeof(int));
//     cudaMalloc(&d_inB, (N+1) * sizeof(char));

//     cudaMemcpy(d_inA, h_inA, N * sizeof(int), cudaMemcpyHostToDevice);
//     cudaMemcpy(d_inB, h_inB, (N+1) * sizeof(char), cudaMemcpyHostToDevice);

//     int *tmpA;
//     char *tmpB;
//     cudaMalloc(&tmpA, N * sizeof(int));
//     cudaMalloc(&tmpB, N * sizeof(char));

//     printf("Initial:\n");
//     printArrayInt("A", h_inA, N);
//     printArrayChar("B", h_inB, N+1);

//     ess_Gpu_test(d_inA, d_inB, N, tmpA, tmpB);

//     cudaMemcpy(h_inA, d_inA, N * sizeof(int), cudaMemcpyDeviceToHost);
//     cudaMemcpy(h_inB, d_inB, N * sizeof(char), cudaMemcpyDeviceToHost);

//     printf("\nFinal:\n");
//     printArrayInt("A", h_inA, N);
//     printArrayChar("B", h_inB, N+1);

//     cudaFree(d_inA);
//     cudaFree(d_inB);
//     cudaFree(tmpA);
//     cudaFree(tmpB);
// }


// end of part for modification
// konec casti k modifikaci

int main( void ) {

  test_repeat(128);

    int N,i;
    cudaEvent_t start, stop;
    float elapsedTime;

    //cudaStream_t stream;
    int *h_A,*h_A2,*h_tA;
    char *h_B,*h_tB;
    int *d_A,*d_tA;
    char *d_B,*d_tB;
    int res,cfg=0;
    char *s;
    char sub1[] = "4070";
    char sub2[] = "A100";
    double limit;

   srand(time(NULL));

      

      cudaDeviceProp prop;
      cudaGetDeviceProperties(&prop, 0);
      printf("Device name: %s\n", prop.name);
          printf("CC: %d.%d\n",  prop.major,prop.minor);  
      printf("#SM: %d\n",  prop.multiProcessorCount);           
    s=prop.name;
    limit=1000.0;
    res = find(s,sub1);
    
    if (res >=0)
        {	cfg=1; limit=87.7; }
    res = find(s,sub2);      
    if (res >=0)
        {	cfg=2; limit=117.6; }

    // start the timers
    HANDLE_ERROR( cudaEventCreate( &start ) );
    HANDLE_ERROR( cudaEventCreate( &stop ) );

    int bar;
    int err=0;
    double tt=0.0;
    for(int mea=0;mea<6;mea++)     
    {
    if (mea<3) N=10'000'000;
    else       N=200'000'000;
    
    int m=mea%3;
    if (m==0) bar=N/30;
    if (m==1) bar=3000;
    if (m==2) bar=10;
    
    HANDLE_ERROR( cudaMalloc( (void**)&d_A, N * sizeof(int) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&d_B, (N+1) * sizeof(char) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&d_tA, N * sizeof(int) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&d_tB, (N+1) * sizeof(char) ) );

    h_A=(int *)malloc(N * sizeof(int));
    h_A2=(int *)malloc(N * sizeof(int));
    h_tA=(int *)malloc(N * sizeof(int));
    h_B=(char *)malloc((N+1) * sizeof(char));
    h_tB=(char *)malloc((N+1) * sizeof(char));
    
    for (i=0; i<N; i++) {
      h_A[i] = 2001-rand()%4001;
      h_B[i]=0;
    }
    h_B[0]=1;
    h_B[N]=1;
    for (i=0; i<bar; i++) {
      h_B[rand()%N]=1;
    }

    /*
    for(int j=0;j<50;j++)
      printf("%i %i\n",j,hostA[j]);
    */
    cudaMemcpy(d_A,h_A,sizeof(int)*N,cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,h_B,sizeof(char)*(N+1),cudaMemcpyHostToDevice);
    for(i=0;i<2;i++)
    {

      if (i!=0) {
        HANDLE_ERROR(cudaDeviceSynchronize());
      }
    HANDLE_ERROR( cudaEventRecord( start, 0 ) );


    switch(i){
    case 0: CPU( h_A, h_B, N );
            break;
    case 1: ess_Gpu(d_A, d_B, N, d_tA, d_tB);
            break;
    }
    if (i!=0) HANDLE_ERROR(cudaDeviceSynchronize());
    HANDLE_ERROR( cudaEventRecord( stop, 0 ) );
    HANDLE_ERROR( cudaEventSynchronize( stop ) );
    HANDLE_ERROR( cudaEventElapsedTime( &elapsedTime, start, stop ) );
    if (i==0) printf( "CPU");
    else      printf( "GPU");
    printf( " time taken:  %g ms\n",elapsedTime );
    
    /*for(int j=0;j<50;j++)
      printf("%i %i %i\n",j,host2[j],hostA[j]);
    */
    if (i==0) continue;
    //printf("before\n");
    //fflush(stdout);
    cudaMemcpy(h_A2,d_A,sizeof(int)*N,cudaMemcpyDeviceToHost);
    //printf("after\n");
    //fflush(stdout);
    for(int j=0;j<N;j++)
      if (h_A2[j]!=h_A[j]) err++;
    printf("Errors=%i\n",err);
    fflush(stdout);
    tt+=elapsedTime;
    if ((err>0)||(tt>800.0))
    {
      printf("Error(s) or too slow (GPU time >800 ms )=> NO points\n");
      break;
    }
    } // i-for
    free(h_A);
    free(h_A2);
    free(h_tA);
    free(h_B);
    free(h_tB);

    HANDLE_ERROR( cudaFree( d_A ) );
    HANDLE_ERROR( cudaFree( d_B ) );
    HANDLE_ERROR( cudaFree( d_tA ) );
    HANDLE_ERROR( cudaFree( d_tB ) );
    if ((err>0)||(tt>800.0))
    {
      break;
    }

    } // mea-for
    if ((err>0)||(tt>800.0))
    {
      //printf("Total GPU time %g err=%d\n",tt,err);
      //printf("0 SJ2 points\n");
      return 0;
    }
    
    if ((cfg==1)||(cfg==2))
    {
    float tmp;
    tmp=12.0*limit/tt;
    printf("Total GPU time %g ms\n",tt);
    if (tmp>15.0) tmp=15.0;
    printf("SJ2 %g points\n",tmp);
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
