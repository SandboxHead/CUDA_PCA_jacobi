#include "lab3_cuda.h"
#include <bits/stdc++.h>
#include <ctime>
#include <ratio>
#include <chrono>

#define CONV_THRESHOLD 1e-3

// #include "jacobi.cpp"
// /*
// 	*****************************************************
// 		TODO -- You must implement this function
// 	*****************************************************
// */
#define TILE_DIM 16

using namespace std;
__global__ void update_pq(int N, int *p, int*q){
	// printf("Updating PQ\n");
	int tid = threadIdx.x;
	int i = blockIdx.x;

	int ind1 = (tid + i)%(N-1);
   	int ind2;
   	if(tid != 0) ind2 = ((N-tid)+i - 1)%(N-1);
   	else ind2 = N - 1;

   	int valp = min(ind1, ind2);
   	int valq = max(ind1, ind2);

   	// printf("%d %d\n", valp, valq);

   	p[i*(N)/2 + tid] = valp;
   	q[i*(N)/2 + tid] = valq;
}

// __global__ void pairscossin(int *N){
//    	// int tid = threadIdx.x + blockIdx.x*blockDim.x;

//    	// int ind1 = (tid + *i)%(*N-1);
//    	// int ind2;
//    	// if(tid != 0) ind2 = ((*N-tid)+*i - 1)%(*N-1);
//    	// else ind2 = *N - 1;

//    	// int row = min(ind1, ind2);
//    	// int col = max(ind1, ind2);
//    	// printf("%d %d\n", row, col);

// 	int tid = threadIdx.x;
// 	int i = blockIdx.x;

// 	int ind1 = (tid + *i)%(*N-1);
//    	int ind2;
//    	if(tid != 0) ind2 = ((*N-tid)+*i - 1)%(*N-1);
//    	else ind2 = *N - 1;

//    	int valp = min(ind1, ind2);
//    	int valq = max(ind1, ind2);

//    	printf("%d %d\n", valp, valq);


// }

__global__ void cossin(int *N, double *D, double *c, double *s, int *pcurr, int *qcurr){
	// printf("Inside cossin\n");
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	// printf("%d\n", tid);
	int row = pcurr[tid];
	int col = qcurr[tid];

	// printf("[@] Inside cossin update: \n" );

	// printf("[@] row col: %d %double\n\n", row, col);

	double p = D[row*(*N) + col];
    double y = (D[col*(*N) + col] - D[row*(*N) + row]) / 2.0;
    double d = fabs(y) + sqrt(p*p + y*y);
    double r = sqrt(p*p + d*d);

    // printf("%f %f %f %f\n", p, y, d, r);

    if(fabs(p) < CONV_THRESHOLD && fabs(d) < CONV_THRESHOLD){
    	c[tid] = 1.0;
    	s[tid] = 0.0;
    }
    else{
	    c[tid] = d / r;
		s[tid] = (fabs(y)/y)*(p / r);
    }

    // printf("%d: %f %f\n\n\n\n",tid, (double)c[tid], (double)s[tid] );
}



__global__ void row_update(int* N, double* D, double* out, double* c, double* s, int* pcurr, int* qcurr){
	__shared__ int p, q;
	__shared__ double co, si;

	if(threadIdx.x == 0){
		p = pcurr[blockIdx.x];
		q = qcurr[blockIdx.x];
		co = c[blockIdx.x];
		si = s[blockIdx.x];
		// printf("[@] Inside Row update: \n" );
		// printf("%d %d %f %f \n", p, q, co, si);
	
	}



	__syncthreads();



	int i = threadIdx.x;


	double val1 = D[p*(*N)+i];
	double val2 = D[q*(*N)+i];



	out[i*(*N)+p] = co*val1 - si*val2;
	// out[i*(*N)+q] = si*val1 + co*val2;

}

__global__ void row_update2(int* N, double* D, double* out, double* c, double* s, int* pcurr, int* qcurr){
	__shared__ int p, q;
	__shared__ double co, si;

	if(threadIdx.x == 0){
		p = pcurr[blockIdx.x];
		q = qcurr[blockIdx.x];
		co = c[blockIdx.x];
		si = s[blockIdx.x];
	}
	__syncthreads();

	int i = threadIdx.x;

	double val1 = D[p*(*N)+i];
	double val2 = D[q*(*N)+i];

	// out[i*(*N)+p] = co*val1 - si*val2;
	out[i*(*N)+q] = si*val1 + co*val2;

}

__global__ void col_update(int* N, double* D, double* out, double* c, double* s, int* pcurr, int* qcurr){
	__shared__ int p, q;
	__shared__ double co, si;

	if(threadIdx.x == 0){
		p = pcurr[blockIdx.x];
		q = qcurr[blockIdx.x];
		co = c[blockIdx.x];
		si = s[blockIdx.x];
	}
	__syncthreads();
	int i = threadIdx.x;

	double val1 = D[p*(*N)+i];
	double val2 = D[q*(*N)+i];

	// double val3 = D1[i*(*N)+p];
	// double val4 = D1[i*(*N)+q];


	out[p*(*N)+i] = co*val1 - si*val2;
	out[q*(*N)+i] = si*val1 + co*val2;

	// out1[i*(*N)+p] = co*val3 - si*val4;
	// out1[i*(*N)+q] = si*val3 + co*val4;


}


__global__ void col_update2(int* N, double* D, double* out, double* c, double* s, int* pcurr, int* qcurr, int N1){
	__shared__ int p, q;
	__shared__ double co, si;

	if(threadIdx.x == 0){
		p = pcurr[blockIdx.x];
		q = qcurr[blockIdx.x];
		co = c[blockIdx.x];
		si = s[blockIdx.x];
	}
	__syncthreads();
	if(*N!=N1 && q == *N-1){
		return ;
	}
	int i = threadIdx.x;

	double val1 = D[i*(*N)+p];
	double val2 = D[i*(*N)+q];

	// double val3 = D1[i*(*N)+p];
	// double val4 = D1[i*(*N)+q];


	out[i*(*N)+p] = co*val1 - si*val2;
	out[i*(*N)+q] = si*val1 + co*val2;

	// out1[i*(*N)+p] = co*val3 - si*val4;
	// out1[i*(*N)+q] = si*val3 + co*val4;


}

__global__ void copy_arr(double* old_arr, double* new_arr){
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	new_arr[tid] = old_arr[tid];
}

__global__ void transpose(double *idata, double *odata, int width, int height)
{
	__shared__ double block[TILE_DIM][TILE_DIM+1];
	
	// read the matrix tile into shared memory
        // load one element per thread from device memory (idata) and store it
        // in transposed order in block[][]
	unsigned int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
	unsigned int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;
	if((xIndex < width) && (yIndex < height))
	{
		unsigned int index_in = yIndex * width + xIndex;
		block[threadIdx.y][threadIdx.x] = idata[index_in];
	}

        // synchronise to ensure all writes to block[][] have completed
	__syncthreads();

	// write the transposed matrix tile to global memory (odata) in linear order
	xIndex = blockIdx.y * TILE_DIM + threadIdx.x;
	yIndex = blockIdx.x * TILE_DIM + threadIdx.y;
	if((xIndex < height) && (yIndex < width))
	{
		unsigned int index_out = yIndex * height + xIndex;
		odata[index_out] = block[threadIdx.x][threadIdx.y];
	}
}

__global__ void MatMul(double* A, double* B, double* C, int ARows, int ACols, int BRows, int BCols, int CRows, int CCols) {

  double CValue = 0;
  int Row = blockIdx.y*TILE_DIM + threadIdx.y;
  int Col = blockIdx.x*TILE_DIM + threadIdx.x;

  __shared__ double As[TILE_DIM][TILE_DIM];
  __shared__ double Bs[TILE_DIM][TILE_DIM];

    for (int k = 0; k < (TILE_DIM + ACols - 1)/TILE_DIM; k++) {

      if (k*TILE_DIM + threadIdx.x < ACols && Row < ARows) As[threadIdx.y][threadIdx.x] = A[Row*ACols + k*TILE_DIM + threadIdx.x];
      else As[threadIdx.y][threadIdx.x] = 0.0;

      if (k*TILE_DIM + threadIdx.y < BRows && Col < BCols)  Bs[threadIdx.y][threadIdx.x] = B[(k*TILE_DIM + threadIdx.y)*BCols + Col];
      else Bs[threadIdx.y][threadIdx.x] = 0.0;

      __syncthreads();

      for (int n = 0; n < TILE_DIM; ++n) CValue += As[threadIdx.y][n] * Bs[n][threadIdx.x];

      __syncthreads();

  }

  if (Row < CRows && Col < CCols) C[((blockIdx.y * blockDim.y + threadIdx.y)*CCols)+(blockIdx.x*blockDim.x)+threadIdx.x]=CValue;

}


bool check_convergence(int N, double* D, double* D_new){
	double sqr_diff = 0;
	for(int i=0; i<N; i++){
		double diff = D_new[i*N + i] - D[i*N + i];
		if(diff<0) sqr_diff -= diff;
		else sqr_diff += diff;
		// if(sqr_diff > CONV_THRESHOLD) return false;
	}
	// double e_sum = 0;
	// // # pragma omp parallel for collapse(2) schedule(static,1) reduction(+:e_sum)
	// for(int i=0; i<N; i++){
	// 	for(int j=0; j<N; j++){
	// 		double diff = E_new[i*N + j] - E[i*N + j];
	// 		if(diff<0) sqr_diff -= diff;
	// 		else sqr_diff += diff;
	// 		if(sqr_diff > CONV_THRESHOLD) return false;
	// 	}
	// }
	// sqr_diff += e_sum;
	// cout << sqr_diff << endl;
	// int i;
	// cin >> i;
	cout << sqr_diff << endl;
	return (sqr_diff < CONV_THRESHOLD);
}

// void mult(int m1, int m2n1, int n2, double* A, double* B, double* C) 
// { 
// 	cout << "Inside" << endl;

// 	double *B_T = (double*)malloc(sizeof(double)*m2n1*n2);
// 	cout << "Inside" << endl;

//     for(int i=0; i<m2n1; i++){
//     	for(int j=0; j<n2; j++){
//     		// cout << "Transposed: " << endl;
    
//     		B_T[j*m2n1 + i] = B[i*n2 + j];
//     	}
//     }
//     // cout << "Transposed: " << endl;
//     for (int i = 0; i < m1; i++)  
//     { 
//         for (int j = 0; j < n2; j++)  
//         { 
//             C[i*n2 + j] = 0; 
//             for (int x = 0; x < m2n1; x++)  
//             { 
//                 C[i*n2 + j] += A[i*m2n1 + x] * B_T[j*m2n1 + x];
//                 // C[i*n2 + j] += A[i*m2n1 + x] * B[x*n2 + j];

//             } 
//         } 
//     }
//     free(B_T);
 
// } 

double check_eigenvals(int N, double* D){
	double out = 0;
	for(int i=0; i<N; i++){
		for(int j=0; j<N; j++){
			if(i!=j){
				out+=fabs(D[i*N + j]);
			}
		}
	} 
	return out;
}

void print_matrix(string name, int M, int N, double* A){
	cout << name << ": \n";
	for(int i=0; i<M; i++){
		for(int j=0; j<N; j++){
			cout << A[i*N + j] << " " ;
		}
		cout << endl;
	}
	cout << endl;
}


void jacobi_parallel(int N, double* D, double* eigenvecs_out, double* eigenvals_out){
	double *eigenvals = eigenvals_out;
	double *eigenvecs = eigenvecs_out;
	int N2 = N;
	double *ENEW;
	double *DNEW;
	if(N%2==1){

		ENEW = (double*)calloc((N+1)*(N+1), sizeof(double));
		DNEW = (double*)calloc((N+1)*(N+1), sizeof(double));
		for(int i=0; i<N; i++){
			for(int j=0; j<N; j++){
				DNEW[i*(N+1) + j] = D[i*N + j];
				ENEW[i*(N+1) + j] = 0;
			}
			ENEW[i*(N+1)+i] = 1;
		}
		D = DNEW;
		N = N+1;
		D[N*N - 1] = 1;
		ENEW[N*N - 1] = 1;
		eigenvals = D;
		eigenvecs = ENEW;
		// printf("Done\n");
	}

	double *dD, *Dtemp, *eignevecs_D, *eignevecs_D_temp;
	std::chrono::high_resolution_clock::time_point t1, t2;
	t1 = std::chrono::high_resolution_clock::now();
	cudaMalloc((void**)&dD, sizeof(double)*N*N);
	cudaMalloc((void**)&Dtemp, sizeof(double)*N*N);
	cudaMalloc((void**)&eignevecs_D, sizeof(double)*N*N);
	cudaMalloc((void**)&eignevecs_D_temp, sizeof(double)*N*N);


	// for(int i=0; i<N*N; i++){
	// 	eigenvals[i] = D[i];
	// }

	cudaMemcpy(dD, D, sizeof(double)*N*N, cudaMemcpyHostToDevice);
	cudaMemcpy(Dtemp, D, sizeof(double)*N*N, cudaMemcpyHostToDevice);
	
	cudaMemcpy(eignevecs_D, eigenvecs, sizeof(double)*N*N, cudaMemcpyHostToDevice);
	cudaMemcpy(eignevecs_D_temp, eigenvecs, sizeof(double)*N*N, cudaMemcpyHostToDevice);

	int *dN, *dp, *dq;
	double *c, *s;
	cudaMalloc((void **)&dN, sizeof(int));
	// cudaMalloc((void **)&di, sizeof(int));
	cudaMalloc((void **)&c, sizeof(double)*N/2);
	cudaMalloc((void **)&s, sizeof(double)*N/2);

	// print_matrix("D", N, N, D);

	cudaMemcpy(dN, &N, sizeof(int), cudaMemcpyHostToDevice);

	double *Dvoidtemp = (double*)malloc(sizeof(double)*N*N);
	double conv = false;
	t2 = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);

  	// std::cout << "[@]Initial Jacobi time: " << time_span.count() << " seconds.\n";

	cudaMalloc((void **)&dp, sizeof(int)*N*(N-1)/2);
	cudaMalloc((void **)&dq, sizeof(int)*N*(N-1)/2);
	update_pq<<<N-1, N/2>>>(N, dp, dq);


	int *p = (int*)malloc(sizeof(int)*N*(N-1)/2);
	int *q = (int*)malloc(sizeof(int)*N*(N-1)/2);

	cudaMemcpy(p, dp, sizeof(int)*N*(N-1)/2, cudaMemcpyDeviceToHost);
	cudaMemcpy(q, dq, sizeof(int)*N*(N-1)/2, cudaMemcpyDeviceToHost);
	

	cudaDeviceSynchronize(); 
	// print_matrix("D", 4, 4, D);
	int sweeps = 0;
	while(!conv){
		t1 = std::chrono::high_resolution_clock::now();

		int N1;

		if(N%2 != 0) N1 = N;
		else N1 = N-1;
		// if(N%2 != 0) N2 = N-1;
		// else N2 = N;
		for(int i=0; i<N1; i++){
			int *currp = dp+(i*(N/2));
			int *currq = dq+(i*(N/2));
			// cudaMemcpy(di, &i, sizeof(int), cudaMemcpyHostToDevice);

			// cudaMemcpy(eigenvals, dD, sizeof(double)*N*N, cudaMemcpyDeviceToHost);
			// print_matrix("eigenvals", N, N, eigenvals);
			// printf("Printing finished\n");

			cossin<<<N/2, 1>>>(dN, dD, c, s, currp, currq);
			cudaDeviceSynchronize();
			// printf("COSSIN updated\n");
			row_update<<<N/2, N>>>(dN, dD, Dtemp, c, s, currp, currq);
			row_update2<<<N/2, N>>>(dN, dD, Dtemp, c, s, currp, currq);
			
			// cudaMemcpy(eigenvals, Dtemp, sizeof(double)*N*N, cudaMemcpyDeviceToHost);
			// print_matrix("After row update", N, N, eigenvals);

			cudaDeviceSynchronize();
		    // dim3 dimBlock(N, 2);

			col_update<<<N/2, N>>>(dN, Dtemp, dD, c, s, currp, currq);
			
			// cudaMemcpy(eigenvals, dD, sizeof(double)*N*N, cudaMemcpyDeviceToHost);
			// print_matrix("After col update", N, N, eigenvals);

			// cudaDeviceSynchronize();
			col_update<<<N/2, N>>>(dN, eignevecs_D, eignevecs_D_temp, c, s, currp, currq);
			cudaDeviceSynchronize();
			copy_arr<<<N, N>>>(eignevecs_D_temp, eignevecs_D);
			cudaDeviceSynchronize();
			// while();

		}
		// while(true){};

		cudaMemcpy(Dvoidtemp, dD, sizeof(double)*N*N, cudaMemcpyDeviceToHost);

		// print_matrix("D", 4, 4, D);

		cout << "Sweep " << ++sweeps << ": ";

		conv = check_convergence(N, eigenvals, Dvoidtemp);
		t2 = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);


		double* tempor = eigenvals;
		eigenvals = Dvoidtemp;
		Dvoidtemp = tempor;

		// dD = Dtemp2;
		double valdiff = check_eigenvals(N, eigenvals);
		// cout << "Valdiff: " << valdiff << endl;
		// std::cout << "[@]Time taken: " << time_span.count() << " seconds.\n";

	}
	// print_matrix("Eigenvals", N, N, eigenvals);
	double* eigenvecs_temp = (double*)malloc(sizeof(double)*N*N);
	cudaMemcpy(eigenvecs_temp, eignevecs_D, sizeof(double)*N*N, cudaMemcpyDeviceToHost);
	for(int i=0; i<N; i++){
		for(int j=0; j<N; j++){
			eigenvecs[j*N+i] = eigenvecs_temp[N*i + j];
		}
	}

	if(N2%2 == 1){
		for(int i=0; i<N-1; i++){
			for(int j=0; j<N-1; j++){
				eigenvecs_out[i*(N-1) + j] = eigenvecs[i*N + j];
				eigenvals_out[i*(N-1) + j] = eigenvals[i*N + j];

			}
		}
		free(eigenvals);
		free(eigenvecs);
	}

	// free(eigenvecs);
	// free(eigenvals);
	free(eigenvecs_temp);

	cudaFree(dD);
	cudaFree(Dtemp);
	cudaFree(dN);
	cudaFree(c);
	cudaFree(s);
	cudaFree(dp);
	// cudaFree(di)
	cudaFree(dq);
	cudaFree(eignevecs_D);
	cudaFree(eignevecs_D_temp);

	// free(Dvoidtemp);


	// print_matrix("D", N, N, D);
	// eigenvals = D;
}

void SVD(int M, int N, double* D, double** U, double** SIGMA, double** V_T)
{



	double* D_T = (double*)malloc(M*N*sizeof(double));
	// cudaMemcpy(D_T, DT_D, sizeof(double)*N*M, cudaMemcpyDeviceToHost);

	// gpu_matrix_transposer
	// t1 = std::chrono::high_resolution_clock::now();
	for(int i=0; i<M; i++){
		for(int j=0; j<N; j++){
			D_T[j*M+i] = D[N*i + j];
		}
	}

	double* DTD = (double*)calloc(N*N*sizeof(double), sizeof(double));
	// printf("Mult args: %d %d %d\n", N, M, N);
	// mult(N, M, N, D_T, D, DTD);	// Make it parallel using cuda


	double *D_D, *DT_D, *DTD_D;
	cudaMalloc((void **)&D_D, sizeof(double)*N*M);
	cudaMalloc((void **)&DT_D, sizeof(double)*N*M);
	cudaMalloc((void **)&DTD_D, sizeof(double)*N*N);

	cudaMemcpy(D_D, D, sizeof(double)*N*M, cudaMemcpyHostToDevice);
	cudaMemcpy(DT_D, D_T, sizeof(double)*N*M, cudaMemcpyHostToDevice);
	// t2 = std::chrono::high_resolution_clock::now();
	// time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);

  	// std::cout << "cuda malloc Time " << time_span.count() << " seconds.\n";


	dim3 dimGrid((N + TILE_DIM - 1) / TILE_DIM, (N + TILE_DIM - 1) / TILE_DIM);
    dim3 dimBlock(TILE_DIM, TILE_DIM);

	MatMul<<<dimGrid, dimBlock>>>(DT_D, D_D, DTD_D, N, M, M, N, N, N);

	cudaMemcpy(DTD, DTD_D, sizeof(double)*N*N, cudaMemcpyDeviceToHost);
	cudaFree(D_D);
	// cudaFree(DT_D);
	cudaFree(DTD_D);


	double* temp = (double*)calloc(N*N, sizeof(double));
	double* E = (double*)calloc(N*N, sizeof(double));
	for(int i=0; i<N; i++){
		E[i*N+i] = 1;
	}
	// print_matrix("E", N, N, E);
	jacobi_parallel(N, DTD, E, temp);
	// print_matrix("Eigenvals", N, N, temp);
	// print_matrix("Eignevecs", N, N, E);

	// printf("Came out\n");
	vector<pair<double, int>> eigenvals;
	for(int i=0; i<N; i++){
		eigenvals.push_back({temp[i*N + i], i});
	}
	sort(eigenvals.begin(), eigenvals.end());
	reverse(eigenvals.begin(), eigenvals.end());

	// printf("Came out\n");

	double* U_T = (double*) calloc(N*N, sizeof(double));
	double* sigma_inv = (double*) calloc(M*N, sizeof(double));

	for(int i=0; i<N; i++){
		for(int j=0; j<M; j++){
			sigma_inv[i*M+j] = 0;
		}
	}
	// printf("Came out\n");

	for(int k=0; k<N; k++){
		int i = eigenvals[k].second;
		(*SIGMA)[i] = (double)sqrt(eigenvals[i].first);	
		sigma_inv[i*N+i] = 1/(double)sqrt(eigenvals[i].first);	
		
		for(int j=0; j<N; j++){
			// U_T[i*N+k] = E[j*N + j];
			// (*U)[j*N+i] = E[j*N+k];
			(*U)[j*N + k] = E[j*N + i];
		}
	}
	for(int i=0; i<N; i++){
		for(int j=0; j<N; j++){
			U_T[i*N + j] = (*U)[j*N + i];
		}
	}
	// printf("Came out\n");

	double* mult_temp = (double*)calloc(M*N, sizeof(double));

	double *sigma_inv_D, *UT_D, *mult_temp_D, *VT_D;

	cudaMalloc((void **)&sigma_inv_D, sizeof(double)*N*M);
	cudaMalloc((void **)&UT_D, sizeof(double)*N*N);
	cudaMalloc((void **)&mult_temp_D, sizeof(double)*M*N);

	cudaMemcpy(sigma_inv_D, sigma_inv, sizeof(double)*N*M, cudaMemcpyHostToDevice);
	cudaMemcpy(UT_D, U_T, sizeof(double)*N*N, cudaMemcpyHostToDevice);

	
	dim3 dimGrid1((M + TILE_DIM - 1) / TILE_DIM, (N + TILE_DIM - 1) / TILE_DIM);
    dim3 dimBlock1(TILE_DIM, TILE_DIM);

	MatMul<<<dimGrid1, dimBlock1>>>(sigma_inv_D, UT_D, mult_temp_D, M, N, N, N, M, N);
	cudaFree(sigma_inv_D);
	cudaFree(UT_D);
	// cudaMemcpy(DTD, DTD_D, sizeof(double)*N*N, cudaMemcpyDeviceToHost);

	cudaMalloc((void**)&VT_D, sizeof(double)*M*M);
	dim3 dimGrid2((M + TILE_DIM - 1) / TILE_DIM, (M + TILE_DIM - 1) / TILE_DIM);
    dim3 dimBlock2(TILE_DIM, TILE_DIM);

	MatMul<<<dimGrid2, dimBlock2>>>(mult_temp_D, DT_D, VT_D, M, N, N, M, M, M);
	cudaMemcpy(*V_T, VT_D, sizeof(double)*M*M, cudaMemcpyDeviceToHost);
	cudaFree(sigma_inv_D);
	cudaFree(UT_D);
	cudaFree(mult_temp_D);
	cudaFree(VT_D);

	free(D_T);
	free(sigma_inv);
	free(U_T);
	// free(Q);
	// free(R);
	free(temp);
	free(E);

	free(mult_temp);

}

void PCA(int retention, int M, int N, double* D, double* U, double* SIGMA, double** D_HAT, int *K)
{

    int i; 
	double ret = (double)retention/100;
    double sum_sigma = 0;
    for(i=0; i<N; i++){
    	sum_sigma += SIGMA[i]*SIGMA[i];
    }
    // print_matrix("SIGMA", 1, N, SIGMA);
    // printf("RET: %f\n", ret);
    // cout << sum_sigma << endl;
    double var = 0;
    for(i=0; i<N; i++){
    	var += (SIGMA[i]*SIGMA[i])/sum_sigma;
    	if(var > ret) {	

    		*K = i+1;
    		break;
    	}
    }
    if(i==N) *K = N;
    // printf("K: %d\n", *K);

    double* UNew = (double*)malloc(sizeof(double)*N*(*K));
    for(int i=0; i<N; i++){
    	for(int j=0; j<*K; j++){
    		UNew[i*(*K) + j] = U[i*N + j];
    	}
    }

    // print_matrix("u_new", N, *K, UNew);

    *D_HAT = (double*)calloc(M*(*K), sizeof(double));
    double *DD, *DD_HAT, *UD;
	cudaMalloc((void **)&DD, sizeof(double)*M*N);
	cudaMalloc((void **)&DD_HAT, sizeof(double)*M*(*K));
	cudaMalloc((void **)&UD, sizeof(double)*N*(*K));

	cudaMemcpy(DD, D, sizeof(double)*M*N, cudaMemcpyHostToDevice);
	cudaMemcpy(UD, UNew, sizeof(double)*N*(*K), cudaMemcpyHostToDevice);

	dim3 dimGrid2((M + TILE_DIM - 1) / TILE_DIM, (M + TILE_DIM - 1) / TILE_DIM);
    dim3 dimBlock2(TILE_DIM, TILE_DIM);

	MatMul<<<dimGrid2, dimBlock2>>>(DD, UD, DD_HAT, M, N, N, *K, M, *K);
	cudaMemcpy(*D_HAT, DD_HAT, sizeof(double)*M*(*K), cudaMemcpyDeviceToHost);

}


void SVD_and_PCA (int M, 
        int N, 
        double* D, 
        double** U, 
        double** SIGMA, 
        double** V_T, 
        int* SIGMAm,
        int* SIGMAn, 
        double** D_HAT, 
        int *K,
        int retention) {

	*SIGMAm = N;
	*SIGMAn = M;
	*U = (double*) malloc(sizeof(double) * N*N);
	*SIGMA = (double*) malloc(sizeof(double) * N);
	*V_T = (double*) malloc(sizeof(double) * M*M);

	SVD(M, N, D, U, SIGMA, V_T);
	PCA(retention, M, N, D, *U, *SIGMA, D_HAT, K);

}


// void read_file(const char* filename, int *num_samples, int *num_features, double** A) {
// 	// std::chrono::high_resolution_clock::time_point t1, t2;
// 	// t1 = std::chrono::high_resolution_clock::now();

//     ifstream ifile;
//     ifile.open(filename, ios::in);
//     int M, N;
//     ifile >> M >> N;
//     cout << M << " " << N << endl;
//     *A = (double *)malloc(sizeof(double)*M*N);
//     num_samples[0] = M;
//     num_features[0] = N;
//     double tmp;
//     for (int i=0; i<M; i++) {
//         for (int j=0; j<N; j++){
//             ifile >> tmp;
//             *((*A) + i*N + j) = tmp;
//         }
//     }

//     ifile.close();

// 	// t2 = std::chrono::high_resolution_clock::now();
// 	// std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);

//   	// std::cout << "File Reading Time " << time_span.count() << " seconds.\n";


// }


// int main(int argc, char* argv[]){
// 	char* filename = argv[1];
// 	int M, N;
// 	// int N = 100;

// 	// double *A = (double*) malloc(sizeof(double)*M*N);
// 	double *A;
// 	read_file(filename, &M, &N, &A);
// 	// double A[16] = {1,2,3,4,
// 	// 			 5,6,7,8,
// 	// 			 2,3,4,5,
// 	// 			 6,7,8,9};

//     // double A[16] = {23, 42, 234, 12,
// 				// 	12, 34, 89, 76,
// 				// 	76, 85, 43, 1,
// 				// 	6, 8, 4, 0};



// 	double *U, *SIGMA, *V_T, *D_HAT;
// 	int K, retention = stoi(argv[2]);

// 	SVD_and_PCA(M, N, A, &U, &SIGMA, &V_T, &D_HAT, &K, retention);
// }


// int main(){
// 	int N = 8;
// 	int *dN;
// 	int *di, *p, *q;

// 	cudaMalloc((void **)&dN, sizeof(int));
// 	cudaMalloc((void **)&di, sizeof(int));
// 	cudaMemcpy(dN, &N, sizeof(int), cudaMemcpyHostToDevice);
// 	// cudaMalloc((void **))
// 	cudaMalloc((void **)&p, sizeof(int)*N*(N-1)/2);
// 	cudaMalloc((void **)&q, sizeof(int)*N*(N-1)/2);

// 	update_pq<<<N-1, N/2>>>(dN, p, q);

// 	int *vp = (int*)malloc(sizeof(int)*N*(N-1)/2);
// 	int *vq = (int*)malloc(sizeof(int)*N*(N-1)/2);

// 	cudaMemcpy(vp, p, sizeof(int)*N*(N-1)/2, cudaMemcpyDeviceToHost);
// 	cudaMemcpy(vq, q, sizeof(int)*N*(N-1)/2, cudaMemcpyDeviceToHost);


// 	for(int i=0; i<N-1; i++){
// 		cout << "[@] " << i << endl;
// 		for(int j=0; j<N/2; j++){
// 			cout << vp[i*N/2 + j] << " " << vq[i*N/2 + j] << endl;
// 		}
// 		cout << endl;
// 	}

// }

