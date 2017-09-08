//**************************************************************
// Assignment #6
// Name: Tejaswi Prakhya
// GPU Programming : Hybrid Programming (An example to implement CUDA + MPI)
// Date of Submission (04/28/2017)
//***************************************************************
// File name : TejaswiPrakhya6.cu
// Compile using : /opt/cuda-8.0/bin/nvcc -I/opt/openmpi/include -L/opt/openmpi/lib -lmpi TejaswiPrakhya6.cu -o exec
// Run using : mpirun --mca mpi_cuda_support 0 ./exec > hw6.txt 
// Host generates random numbers in an array of size as required. (Please modify the value of N, as required)
// Kernel will Check if each number is even or odd in generated array. That is written in CUDA
// Each process returns even numbers and odd numbers from an array which was written in MPI.
//*****************************************************************


//******************************************************************
//Header files
//*******************************************************************

#include<mpi.h>
#include<cuda.h>
#define N 1024
#include<stdlib.h>
#define T 1024 // max threads per block
#include <stdio.h>

//====================================================
/*             Kernel Declaration                  */
__global__ void check_num (int *dev_a, int *dev_b, int *dev_c);


//====================================================
/*             Main Function                       */

int main() {

	int a[N], b[N], c[N];								// Declaed arrays in HOST
	int *dev_a, *dev_b, *dev_c;							// Pointers to DEVICE locations
	
	// initialize a and b with real values "NOT SHOWN" For students to do at home
	int size = N * sizeof(int);							// Allocating size of an array
	
	cudaMalloc((void**)&dev_a, size);						// Reserving the right amount of space in the device for array's a,b, and c
	cudaMalloc((void**)&dev_b, size);
	cudaMalloc((void**)&dev_c, size);
       
	
        for(int i= 0; i< N; i++)							// An array that generates random array of range (1,1024) apart from 0
        {
        a[i] = rand()%1024+1;
	//printf("Array at %d is %d", i,a[i]); 
	
	}
	
        cudaMemcpy(dev_a, a, size,cudaMemcpyHostToDevice);				// copy array a to the device Array's dev_a
		
	check_num<<<1,1024>>>(dev_a,dev_b,dev_c);					// Invoked kernel, where the blcok size is 1 and no.of threads is 1024
	
	cudaMemcpy(b, dev_b, size,cudaMemcpyDeviceToHost);				// copy the even and odd number array's collected from GPU (Device) array dev_c,dev_b to CPU-Mem (Host) array c and b respectively
	cudaMemcpy(c, dev_c, size,cudaMemcpyDeviceToHost);
	
	int rank,comm_sz; 								//Rank gives the current process id, comm_sz gives total number of processes
	int count,i,k,z,temp1,temp2;  
	
    	MPI_Init(NULL,NULL);  								// MPI_Initialize, from where MPI code starts
	MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	printf( "Hello, Am processS %d of %d worked to print even and odd numbers \n", rank, comm_sz );

	count = N/comm_sz; 								//To calculate the number of elements that must be processed by each Process(rank)
	if(rank==0)
	{
	
	temp1=1;									//used temporary variables to send process signals from processes apart from Process Id : 0. Loop iterates when process id reaches comm_sz
	temp2=1;
	for(k=count;k<N;k+=count)							// The loop iterates till all the process gets equal number of elements
        {
            MPI_Send(b+k,count,MPI_INT,temp1,0,MPI_COMM_WORLD);				// Distributes the data that is individual even array equally to each process
	    MPI_Send(c+k,count,MPI_INT,temp2,0,MPI_COMM_WORLD);				// Distributes the data that is individual  odd array equally to each process
            temp1++;
	    temp2++;
        }
	

	for(i=0;i<count;i++)								// Printing the even numbers from even array when rank is 0
	{
		if(b[i]!=1)
			printf("The number in even array is %d \n",b[i]);
		if(c[i]!=0)								// Printing the odd numbers from odd array when rank is 0
			printf("The number in even array is %d \n",c[i]);
	}

	}
	
											// When rank !=0, this else part gets executed
	else
	{	
	int even_data[count]; // array to save event numbers count apart from 0 process	// Temporary array to store even array
        MPI_Recv(even_data,count,MPI_INT,0,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);		// each process receives even array 
	int odd_data[count]; // array to save odd numbers count apart from 0 process
        MPI_Recv(odd_data,count,MPI_INT,0,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);		// each process receives odd array
        for(z=0;z<count;z++)
        {
		if((*(even_data+z))!=1)
		{
		printf("The number in even array is %d \n",*(even_data+z)) ;		// Checks for even number and prints it
		}
		if((*(odd_data+z))!=0)
		{
		printf("The number in odd array is %d \n",*(odd_data+z)) ;		// Checks for odd number and prints it
		}
        }
	}
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
	MPI_Finalize(); 								// End of MPI code
	
	exit (0);
}


//====================================================
/*             Kernel Definition                 */
//====================================================

__global__ void check_num (int *dev_a, int *dev_b, int *dev_c) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;				// Generates unique threadId
	dev_b[id]=1;								// Assigning the dev_b array(even array) with all 1's initially
	dev_c[id]=0;								// Assigning the dev_c array(even array) with all 0's initially
	if (id < N) {

		if(dev_a[id] %2 == 0)						// Checks for each number in an initially generated array 'dev_a', if its even or odd
		{
			dev_b[id] = dev_a[id];					// If it's even, saving the even value to dev_b array
		
		}
		else
		{
		
		dev_c[id] = dev_a[id];						// If it's odd, saving the odd value to dev_b array
		
		}
		
	}
}


