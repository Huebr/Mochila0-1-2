
#ifndef __CUDACC__
#define __CUDACC__
#endif
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include <string.h>

//#define WEI 11
//#define ITEN 5

//void inserirPeso(int vet[]);
//void inserirValor(int vet[]);


void info(int tam, int pes[], int val[], int n);
void encItens(int *mol, int n, int W, int pes[], int val[], int nit, int wit);

cudaError_t mochilaWithCuda(int *mochila, const int *peso, const int *valor, int iten, int pes);

__global__ void mochilaKernel(int *mochila, const int *peso, const int *valor,const int wei,const int iten)
{
	int i;
	int w = threadIdx.x + 1;
	if (w<wei+1){
		for (i = 1; i<iten + 1; i++){
			if (peso[i]>w){
				mochila[i*(wei + 1) + w] = mochila[(i - 1)*(wei + 1) + w];

			}
			else{
				mochila[i*(wei+1)+w] = max(mochila[(i - 1)*(wei+1)+w], valor[i] + mochila[(i - 1)*(wei+1)+ w - peso[i]]);

			}
		}
	}
}

int main()
{
    
	char url[80];

	//declaração do peso limite da mochila, o numero de itens e variavel auxiliar
	int W, n, aux, aux2, q, i, j, *peso, *valor, *mochila;

	//C:/Users/Pedro/Downloads/test.in

	//---------------------Leitura do arquivo-----------------------------------------------

	printf("Digite o local do arquivo: ");
	printf("\n");
	scanf("%s", &url);

	FILE *arq;

	arq = fopen(url, "r");
	if (arq == NULL)
		printf("Erro, nao foi possivel abrir o arquivo\n");
	else{
		fscanf(arq, "%d\n", &n);
		aux = n + 1;

		//Alocação
		peso = (int*)malloc(aux*sizeof(int));
		valor = (int*)malloc(aux*sizeof(int));
		if (peso == NULL || valor == NULL){
			perror("Erro de alocacao de memoria vetor peso ou vetor valor");
			exit(EXIT_FAILURE);
		}

		for (i = 1; i<aux; i++){
			fscanf(arq, "%d %d %d\n", &q, &valor[i], &peso[i]);
		}

		fscanf(arq, "%d\n", &W);
		aux2 = W + 1;

		mochila = (int*)malloc(aux*aux2*sizeof(int));
		
		if (mochila == NULL){
			perror("Nao foi possivel alocar a mochila");
			exit(EXIT_FAILURE);
		}
		for (i = 0; i < (aux)*(aux2); i++) mochila[i] = 0;

	}

	//Fecha o Arquivo
	fclose(arq);
	//------------------------Final do Arquvio------------------------------------------


	

	//inserções dos itens com seus respectivos valores
	//inserirPeso(peso);
	//inserirValor(valor);

	info(W, peso, valor, n);

	//Pseudo Codigo Transcrito


    // Add vectors in parallel.
    cudaError_t cudaStatus = mochilaWithCuda(mochila,peso,valor,n,W);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

   

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

	printf("\n");

	encItens(mochila, n, W, peso, valor, n, W);

	printf("\n");
	printf("\n");

	//Imprimindo a matriz
	for (i = 0; i<n + 1; i++) {
		for (j = 0; j<W + 1; j++) {
			printf("%d ", mochila[(i*(W + 1)) + j]);

		}
		printf("\n"); // para pular linha quando terminar a coluna
	}


	printf("\n");


	printf("\n");
	printf("Valor maximo da mochila: %d\n", mochila[n*(W + 1) + W]);

	free(mochila);
	free(valor);
	free(peso);
	system("pause");
    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t mochilaWithCuda(int *host_mochila, const int *host_peso, const int *host_valor, int iten, int pes)
{
    int *dev_mochila = 0;
    int *dev_peso = 0;
    int *dev_valor = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_mochila,(iten + 1)*(pes + 1)*sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_peso, (iten+1) * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_valor, (iten+1) * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_mochila, host_mochila, (iten + 1)*(pes + 1)* sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
	
    cudaStatus = cudaMemcpy(dev_valor, host_valor, (iten+1) * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

	cudaStatus = cudaMemcpy(dev_peso, host_peso , (iten+1) * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}


    // Launch a kernel on the GPU with one thread for each element.
    mochilaKernel<<<1, pes>>>(dev_mochila, dev_peso, dev_valor, pes, iten);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(host_mochila, dev_mochila, (iten+1)*(pes+1)* sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_mochila);
    cudaFree(dev_valor);
    cudaFree(dev_peso);
    
    return cudaStatus;
}

void inserirPeso(int vet[]){
	vet[1] = 1;
	vet[2] = 2;
	vet[3] = 5;
	vet[4] = 6;
	vet[5] = 7;
}

void inserirValor(int vet[]){
	vet[1] = 1;
	vet[2] = 6;
	vet[3] = 18;
	vet[4] = 22;
	vet[5] = 28;
}






void info(int tam, int pes[], int val[], int n){
	int i;
	printf("=========================================================\n");
	printf("                  *Dados da mochila*                     \n");
	printf("\n");
	printf("Capacidade total da mochila: %d\n", tam);
	printf("Numero de itens: %d itens", n);
	printf("\n");

	printf("Valor de cada item: ");
	for (i = 1; i<n + 1; i++){
		printf("%d ", val[i]);
	}

	printf("\n");

	printf("Peso de cada item: ");
	for (i = 1; i<n + 1; i++){
		printf("%d ", pes[i]);
	}



	printf("\n=========================================================\n");
}

void encItens(int *mol, int n, int W, int pes[], int val[], int nit, int wit){

	int i, pcal;


	//mat[n] abc

	if (mol[(n*(wit+1))+W] == mol[((n - 1)*(wit+1))+W]){
		if (mol[(n*(wit+1)) + W] != 0){
			encItens(mol, n - 1, W, pes, val, nit, wit);
		}
	}
	else{

		for (i = 1; i<nit + 1; i++){
			pcal = W - pes[i];
			if (val[i] + mol[((i - 1)*(wit+1)) + pcal] == mol[(n*(wit+1)) + W]){
				printf("%d ", i);
				encItens(mol, i - 1, pcal, pes, val, nit, wit);
				break;
			}
			if (mol[(n*(wit+1)) + W] == val[i]){
				printf("%d ", i);

				break;
			}


		}

	}



}
