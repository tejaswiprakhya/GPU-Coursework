//In this program we tried to generate edge weights in random 
//to avoid user entering edge weights
#include<stdio.h>
#include<time.h>
#include<stdlib.h>
#include "timer.h"

#define max 32767

//function protype
void prims(int , int *);

/****************************************************************************************/
//			Function Name : main
//		In this function we will read input from user and we call the prims function 
/***************************************************************************************/
void main() {
	
	//variable decleration
	int *adj_Mat;
	int n, i, j, temp;
	double timeCompStart, timeCompEnd, timeComp;
	//reading input
	printf("Enter no .of nodes");
	scanf("%d", &n);
	srand(time(NULL));
	//Allcoating memory for graph dynamically
	adj_Mat = malloc(n * n * sizeof(int));
	
	for (i = 0; i < n; i++) {
		*(adj_Mat + i * n + i) = 0;//No self loops in Graph 
		for (j = i + 1; j < n; j++) {
			//generating random number for weight of edge from 0 to 25 
			temp = rand() % n + 0;
			/*printf("weight from node %d --> %d", i + 1, j + 1);
			scanf("%d", (adj_Mat + i * n + j));*/
			//creating adjacency matrix
			*(adj_Mat + j*n + i) = *(adj_Mat + i * n + j) = temp;
			if (*(adj_Mat + i*n + j) == 0) {
				*(adj_Mat + j*n + i) = *(adj_Mat + i*n + j) = max;	//Assigning minimum value max, whose weight is 0.
				
			}
		}
	}
	/*printing Graph
	printf("\n");
	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++) {
			if (*(adj_Mat + i * n + j) == max)
				printf("\t0");
			else
				printf("\t%d", *(adj_Mat + i * n + j));
		}
		printf("\n");
	}*/
	GET_TIME(timeCompStart);
	//function call
	prims(n, adj_Mat);
	GET_TIME(timeCompEnd);
	timeComp = timeCompEnd - timeCompStart;
	printf("\nComputation Time : %f seconds\n",timeComp);
}

/********************************************************************************/
//			Function Name : prims
//			Arguments : adj_Mat(Graph )and n (no.of nodes)
//			This function is used to find the shortest path to travel all nodes in
//			a graph using prims algorithm and it prints each edge in path and it's
//			weight and the total cost of path 
/********************************************************************************/
void prims(int n, int *adj_Mat) {
	
	int *visited;
	int a, b, u, v,i, j, min = max, min_weight = 0, no_edges = 1;
	
	//array to store all visited nodes in graph
	visited = malloc(n * sizeof(int));
	//printf("%d", n);
	//all nodes are not visited
	for (i = 0; i < n; i++) {
		visited[i] = 0;
	}
	//first node is visited
	visited[0] = 1;
	while (no_edges < n) {
		for (i = 0, min = max; i < n; i++) {
			for (j = 0; j < n; j++) {
				if (*(adj_Mat + i*n + j) < min) {
					if (visited[i] != 0) {
						min = *(adj_Mat + i*n + j);
						a = u = i;
						b = v = j;
					}
				}
			}
		}
		if (visited[u] == 0 || visited[v] == 0) {
			//printf("\n Edge No %d  %d ----> %d weight is = %d", no_edges, a + 1, b + 1, min);
			min_weight += min;
			no_edges++;
			visited[b] = 1;
		}
		//visited edge weight is changed to max weight
		*(adj_Mat + u * n + v) = *(adj_Mat + v * n + u) = max;
	}
	//printing total cost
	printf(" \n Minimum Cost = %d", min_weight);
}