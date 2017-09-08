// ****************************************************************************************************
//   ASSIGNMENT 1:  CUDA program to compute the area under the curve by using the Trapezoid Approach.
//   Name        :  Tejaswi Prakhya
// ****************************************************************************************************

#define T 1024 											// Maximum number of threads per block
#include <stdio.h>
#include <math.h>

//******************KERNEL DECLARATION*****************************************

__global__ void Area_one (float *, float *, float *, float *);
__global__ void Area_four (float *, float *, float *, float *);
//*****************************************************************************

/*             Main Function                       */

int main() 
	{

		float a, b;									//Declaring the variables in Host a : Starting Interval,b : Ending Interval,Number of Trapezoids :n 
		float n;
		float *dev1_a, *dev1_b, *dev1_height;						//Declaring pointers to device locations
		float *dev1_n;
		float *dev1_res;
		float *dev2_a, *dev2_b, *dev2_height, *dev2_n, *dev2_res;
		printf("Enter value of a : ");							//Endusers enters the value for a,b,n
        	scanf ("%f", &a);
        	printf("Enter value of b : ");
        	scanf ("%f", &b);
		printf("Enter value of n - no of trapezoids :");
        	scanf ("%f", &n);
        	float height = (b-a)/n;								//Calculating height of each trapezoid as per Trapezoidal rule
		//printf("%f\n", height);
	
		float size = T * sizeof(float);							//Memory allocated to handle 1024 threads (1 block)
		float size2 = 4*T *sizeof(float);						//Memory allocated to handle 4*1024 threads (4 blocks)
		float* result1 = (float*)malloc(size);						//Reserving the space to store values calculated for y= f(X*x) for respective intervals 
		float* result2 = (float*)malloc(size2);
		
		//Reserving the space in device 
		
		cudaMalloc((void**)&dev1_a, size);
		cudaMalloc((void**)&dev1_b, size);
		cudaMalloc((void**)&dev1_n, size);
        	cudaMalloc((void**)&dev1_res, size);
		cudaMalloc((void**)&dev1_height, size);

		cudaMalloc((void**)&dev2_a, size2);
		cudaMalloc((void**)&dev2_b, size2);
		cudaMalloc((void**)&dev2_n, size2);
        	cudaMalloc((void**)&dev2_res, size2);
		cudaMalloc((void**)&dev2_height, size2);
		
        	//Copying information from Host to Device
        
		cudaMemcpy(dev1_a, &a, size,cudaMemcpyHostToDevice);
		cudaMemcpy(dev1_b, &b, size,cudaMemcpyHostToDevice);
		cudaMemcpy(dev1_n, &n, size,cudaMemcpyHostToDevice);
		cudaMemcpy(dev1_height, &height, size,cudaMemcpyHostToDevice);
		
		cudaMemcpy(dev2_a, &a, size,cudaMemcpyHostToDevice);
		cudaMemcpy(dev2_b, &b, size,cudaMemcpyHostToDevice);
		cudaMemcpy(dev2_n, &n, size,cudaMemcpyHostToDevice);
		cudaMemcpy(dev2_height, &height, size,cudaMemcpyHostToDevice);
		
		//Invoking the Kernel for 1 block, 1024 Threads
		
		Area_one<<<1,T>>>(dev1_a,dev1_n,dev1_height,dev1_res);

		//Invoking the Kernel for 4 block, 1024 Threads
	
		Area_four<<<4,T>>>(dev2_a,dev2_n,dev2_height,dev2_res);
	
		//Copying the Result from Device to Host that calculated the area for y=f(X*X) at each interval
		
		cudaMemcpy(result1,dev1_res, size,cudaMemcpyDeviceToHost);
		
		cudaMemcpy(result2,dev2_res, size2,cudaMemcpyDeviceToHost);
		
		//printf("%d\n", result2[2049]);
		
		//Calculating total sum of area's that are calculated for 'n' trapezoids at each interval.
		float sum1 = 0.0;
		for(int i=0; i<T; i++) 
			{
				sum1 += result1[i];
			}
		
		float sum2 = 0.0;
		for(int i=0; i<4*T; i++) 
			{
				sum2 += result2[i];
			}
		//Multiplying the above calculated total sum with height/2
		
		sum1 *= height/2.0;
		sum2 *= height/2.0;
		
		//Free device memory

		cudaFree(dev1_a);
		cudaFree(dev1_b);
		cudaFree(dev1_n);
		
		cudaFree(dev2_a);
		cudaFree(dev2_b);
		cudaFree(dev2_n);
		
		//Final Area under the given curve calculated using Trapezoidal Approach.
        
        	printf("For the interval [ %f, %f ] and %f trapezoids, the area under the curve for one block is %f \n", a,b,n,sum1);
       		printf("For the interval [ %f, %f ] and %f trapezoids, the area under the curve for four blocks is %f \n", a,b,n,sum2);  
		exit (0);
	}


/*---------------------------------------------------------------------
 * Function:   myfunction
 * Purpose:  Curve for which area should be calculated. y = f(X*X) 
 * In args:  a
 * Out arg:  returns a*a
 */

__device__ float myfunction(float a)
	{

	return a*a;											
	}
/*             Kernel Definition                  */

/*---------------------------------------------------------------------
 * Kernel:   Area_one
 * Purpose:  Implements Sum of trapezoids below the curve
 * In args:  a(starting index),n (number of trapezoids),height (calculated height based on a,b,n) and dev1_res(array to store sum)
 * Out arg:  returns dev1_res 
 */

__global__ void Area_one(float *a,float *n,float *height,float *dev1_res)
	{

		int idx = threadIdx.x;									//calculates ThreadId of one block in one dimension 
        	float temp1_a,temp1_b;
		//Each thread 'idx' will perform 'count' no.of threads work. To handle any multiples of 1024 threads when 1 Block of kernel is invoked.
		
		float count = ((*n)/T);
		float id ;
		
		//Iterating each global threadid work for 'count' no of times.
		temp1_a= *a+(count*idx*(*height));
		temp1_b= temp1_a+(count*(*height));

		dev1_res[idx]=(myfunction(temp1_a)+myfunction(temp1_b));				//calculates sum of area's of first and last trapezoid in calculated count 
	
		for(int i=1; i< count; i++)
			{
				id=temp1_a+(i*(*height));
				dev1_res[idx] += 2* myfunction(id);					//calculates sum of area's of trapezoids between first and last.
	
			}


	}
	
/*---------------------------------------------------------------------
 * Kernel:   Area_four
 * Purpose:  Implements Sum of trapezoids below the curve
 * In args:  a(starting index),n (number of trapezoids),height (calculated height based on a,b,n) and dev1_res(array to store sum)
 * Out arg:  returns dev1_res 
 */
__global__ void Area_four(float *a,float *n,float *height,float *dev2_res)
	{
		
		int idx = blockIdx.x * blockDim.x + threadIdx.x;					//calculates the GlobalThreadId for 4 blocks 
        	float temp2_a,temp2_b;
		//Each thread 'idx' will perform 'count' no.of threads work. To handle any multiples of 4.0*1024 threads when 4 Block of kernel is invoked.

		float count = ((*n)/(4.0*T));
		float id ;
		//Iterating each global threadid work for 'count' no of times.
		temp2_a= *a+(count*idx*(*height));
		temp2_b= temp2_a+(count*(*height));
		
		dev2_res[idx]=(myfunction(temp2_a)+myfunction(temp2_b));				//calculates sum of area's of first and last trapezoid in calculated count 
	
		for(int i=1; i< count; i++)
			{
				id=(temp2_a)+(i*(*height));
				dev2_res[idx] += 2* myfunction(id);					//calculates sum of area's of trapezoids between first and last.
	
			
			}
		}

