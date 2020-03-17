#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
 #include<time.h>

#define BLOCK_SIZE  32

__global__ void gpu_matrix_mult(long *a, long *b, long *c, int m, int n, int k)
{	
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	
	int sum = 0;
	if (row < m)
	{
		if(col < k) {
				for (int i = 0; i < n; i++)
				{
					sum += a[row * n + i] * b[i * k + col];
				}
				c[row * k + col] = sum;
		}
	}
}

void cpu_matrix_mult(long *h_a, long *h_b, long *h_result, int m, int n, int k) {
	for (int i = 0; i < m; ++i)
	{
		for (int j = 0; j < k; ++j)
		{
			int tmp = 0.0;
			for (int h = 0; h < n; ++h)
			{
				tmp += h_a[i * n + h] * h_b[h * k + j];
			}
			h_result[i * k + j] = tmp;
		}
	}
}
void DisplayMatrix(long * h_a , int m,int n)
{
printf("\n");
	for (int i = 0; i < m; ++i)
	{
		for (int j = 0; j < n; ++j)
		{
			printf("%d  ", h_a[i*n + j]);
		}

		printf("\n");
	}
printf("\n");
}

int main(int argc, char const *argv[])
{


  long *dev_a, *dev_b, *dev_c,*dev_d;
  float ms;
  clock_t startc, end;
  double cpu_time_used;
  cudaEvent_t start,stop;
	FILE *myFile_mnk;
    myFile_mnk = fopen("C:\\Users\\k1636\\Documents\\Visual Studio 2010\\Projects\\abc\\abc\\mnk.txt", "r");

    
    int i,j;
	int m,n,k;
	fscanf(myFile_mnk, "%d", &m);
	fscanf(myFile_mnk, "%d", &n);
	fscanf(myFile_mnk, "%d", &k);
	fclose(myFile_mnk);
	printf("%d %d %d",m,n,k);

 long *a=(long*)malloc(m*n*sizeof(int));
long *b=(long*)malloc(n*k*sizeof(int));
long *c=(long*)malloc(m*k*sizeof(int));
long *d=(long*)malloc(m*k*sizeof(int));


 FILE *myFile_input1;
    myFile_input1 = fopen("C:\\Users\\k1636\\Documents\\Visual Studio 2010\\Projects\\abc\\abc\\input1.txt", "r");
    for (i = 0; i < m; i++)
    {	
    	for(j=0;j<n;j++){
		fscanf(myFile_input1, "%d", &a[i*n+j]);
		
		}
        
    }
    fclose(myFile_input1);
    FILE *myFile_input2;
    myFile_input2 = fopen("C:\\Users\\k1636\\Documents\\Visual Studio 2010\\Projects\\abc\\abc\\input2.txt", "r");
    for (i = 0; i < n; i++)
    {	
    	for(j=0;j<k;j++){
		fscanf(myFile_input2, "%d", &b[i*k+j]);
		
		}
        
    }
    fclose(myFile_input2);

printf("\nA\n");
DisplayMatrix(a,m,n);
printf("\nB\n");
DisplayMatrix(b,n,k);






 cudaMalloc((void **) &dev_a, m*n*sizeof(int));
 cudaMalloc((void **) &dev_b, n*k*sizeof(int));
 cudaMalloc((void **) &dev_c, m*k*sizeof(int));
 cudaMalloc((void **) &dev_d, m*k*sizeof(int));

 startc = clock();
 cpu_matrix_mult(a, b, c,  m, n, k) ;
 end = clock();
printf("\n\n CPU RESULT \n\n");

DisplayMatrix(c,m,k);


cpu_time_used = ((float) (end - startc)) /(float) CLOCKS_PER_SEC;
cpu_time_used*=1000;


cudaMemcpy(dev_a, a, m*n*sizeof(int),
cudaMemcpyHostToDevice);
cudaMemcpy(dev_b, b, n*k*sizeof(int),
cudaMemcpyHostToDevice);
cudaMemcpy(dev_d, d, m*k*sizeof(int),
cudaMemcpyHostToDevice);


cudaEventCreate(&start);
cudaEventCreate(&stop);
cudaEventRecord(start, 0); 
cudaEventRecord(stop, 0);



dim3 dimGrid ((k + BLOCK_SIZE - 1) / BLOCK_SIZE, (m + BLOCK_SIZE - 1) / BLOCK_SIZE);

gpu_matrix_mult<<<dimGrid,dim3(BLOCK_SIZE,BLOCK_SIZE)>>>(dev_a, dev_b, dev_c,m,n,k);

cudaEventSynchronize(stop);
cudaEventElapsedTime(&ms, start, stop);
printf("\n\n GPU RESULT \n\n");
cudaMemcpy(d, dev_c, m*k*sizeof(int),cudaMemcpyDeviceToHost);
DisplayMatrix(d,m,k);

cudaEventDestroy(start);
cudaEventDestroy(stop);




printf("GPU: %f ms",ms);
printf("\n CPU : %f ms",cpu_time_used);
system("pause");
 return 0;
}