#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cooperative_groups.h>

#define BLOCK_SIZE 4

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

// --------------------------------------------
// helper: block scan (Blelloch style, simplified)
// --------------------------------------------

// Grenzen müssen weiter gegeben werden!
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
// --------------------------------------------
// Kernel 1: local segmented scan + block sums
// --------------------------------------------
__global__ void kernel1(int* A, char* B, int N, int* blockSum, int* overhead_index, int* lvl, int* block_size, bool* done) {

  extern __shared__ unsigned char shared[];
  __shared__ int pos_first;

  int* sA = (int*)shared;
  int* tmp = (int*)&sA[blockDim.x];
  char* sB = (char*)&tmp[blockDim.x];
  char* tmpb = (char*)&sB[blockDim.x];   // ✅ FIX (war vorher falsch!)

  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  int tid = threadIdx.x;

  if(*lvl == 1){
    // load
    if (gid < N) {
        sA[tid] = A[gid];
        sB[tid] = B[gid];
    } else {
        sA[tid] = 0;
        sB[tid] = 1;
    }
  }else{
    // load
    if (gid < N) {
        sA[tid] = blockSum[gid];
        if(overhead_index[gid] == 20000){
            sB[tid] = 0;
        }else{
            sB[tid] = 1;
        }
    } else {
        sA[tid] = 0;
        sB[tid] = 1;
    }
  }

  // ✅ INIT (SEHR WICHTIG!)
  if (tid == 0) pos_first = 20000 ;
  __syncthreads();

  

  __syncthreads();

  // scan
  segmented_scan_block(sA, tmp, tmpb, sB, blockDim.x);

  __syncthreads();

  // write back
  if (gid < N) {
    if(*lvl == 1){
        if (B[gid + 1]) {
            A[gid] = sA[tid];
        } else {
            A[gid] = 0;
        }
    }else{
        if(overhead_index[gid+1] != 20000){
            A[pow(gid, *lvl)+overhead_index[gid+1]] = sA[tid];
        }
    }
  }

  if (gid < N && B[gid+1]){
    atomicMin(&pos_first,tid*pow(*block_size, *lvl)+sB[gid+1]);
  }
  __syncthreads();


  // 👉 Ergebnis speichern
  if (tid == blockDim.x - 1) {

    overhead_index[blockIdx.x] = pos_first;      // -1 wenn kein Bit gesetzt

    if (gid < N && !B[gid + 1]) {
      blockSum[blockIdx.x] = sA[tid];
    } else {
      blockSum[blockIdx.x] = 0;
    }
  }
}

// --------------------------------------------
// Kernel 2: scan block sums (simple sequential
// because #blocks is small)
// --------------------------------------------
__global__ void kernel2(int* blockSum, int* blockFlag) {


}




__global__ void kernel2_v1(int* blockSum, int* blockFlag, int block_amount) {
  int sum = 0;
  for(int ind = 0; ind < block_amount-1; ind ++){
    sum += blockSum[ind];
    if(blockFlag[ind+1] == 20000){
      blockSum[ind] = 0;
    }else{
      blockSum[ind] = sum;
      sum = 0;
    }

  }
}
// --------------------------------------------
// Kernel 3: add block offsets
// --------------------------------------------
__global__ void kernel3(int* A, char* B, int N, int* blockSum, int blocksize, int* blockFlag) {


  int gid = blockIdx.x * blocksize;

  if (gid >= N) return;

  if (blockFlag[blockIdx.x+1] < 20000) {
    A[gid+blocksize+blockFlag[blockIdx.x+1]] += blockSum[blockIdx.x];
  }
}

// --------------------------------------------
// main GPU wrapper
// --------------------------------------------
void ess_Gpu(int* inA, char *inB, int N, int *tmpA, char *tmpB)
{
  int blockSize = BLOCK_SIZE;
  int numBlocks = (N + blockSize - 1) / blockSize;

  int *blockSum;
  int *blockFlag;

  cudaMalloc(&blockSum, numBlocks * sizeof(int));
  cudaMalloc(&blockFlag, numBlocks * sizeof(int));

  size_t sharedSize = blockSize * (sizeof(int)*2 + sizeof(char)*2);

  // ----------------------------
  // Events erstellen
  // ----------------------------
  cudaEvent_t k1_start, k1_stop;
  cudaEvent_t k2_start, k2_stop;
  cudaEvent_t k3_start, k3_stop;

  cudaEventCreate(&k1_start);
  cudaEventCreate(&k1_stop);
  cudaEventCreate(&k2_start);
  cudaEventCreate(&k2_stop);
  cudaEventCreate(&k3_start);
  cudaEventCreate(&k3_stop);

  // ----------------------------
  // Kernel 1
  // ----------------------------
  cudaEventRecord(k1_start);
  kernel1<<<numBlocks, blockSize, sharedSize>>>(inA, inB, N, blockSum, blockFlag);
  cudaDeviceSynchronize();
  cudaEventRecord(k1_stop);

}


int main( void ) {

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

  memcpy(h_tA, h_A, N * sizeof(int));
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
  int maxPrint = 20;   // wie viele Fehler maximal anzeigen
  int printed = 0;

  for(int j=0; j<N; j++) {
    if (h_A2[j] != h_A[j]) {
      err++;

      if (printed < maxPrint) {
        printf("\n❌ Error at index %d:\n", j);

        // Kontext anzeigen (±3 Elemente)
        int start = (j - 20 < 0) ? 0 : j - 20;
        int end   = (j + 1 >= N) ? N - 1 : j + 1;

        printf("Index : ");
        for(int k=start; k<=end; k++) printf("%6d", k);

        printf("\nInput : ");
        for(int k=start; k<=end; k++) printf("%6d", h_tA[k]);

        printf("\nCPU   : ");
        for(int k=start; k<=end; k++) printf("%6d", h_A[k]);

        printf("\nGPU   : ");
        for(int k=start; k<=end; k++) printf("%6d", h_A2[k]);

        printf("\nB     : ");
        for(int k=start; k<=end; k++) printf("%6d", h_B[k]);

        printf("\n--------------------------\n");

        printed++;
      }
    }
  }

  printf("Total Errors = %i\n", err);
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

