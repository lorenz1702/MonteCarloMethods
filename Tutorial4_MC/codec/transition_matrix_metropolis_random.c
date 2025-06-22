#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define size 5

double minimum(double a, double b){
	if( a > b )
		return b;
	else
		return a;
}

int mul(double *T, double *v, double *u){

	int i,k;

	for(i = 0; i < size; i++){
	
		u[i] = 0;

		for(k = 0; k < size; k++){

			u[i] += T[k*size+i] * v[k];

		}
	}

	return 1;
}

double observable(double *v){

	double o = 0;

	for(int i = 0; i < size; i++){
		o += v[i]*i;
	}

	return o;

}


int main(void){

	double T[size*size];

	for(int i = 0; i < size; i++){
		for(int j = 0; j < size; j++){
			T[i*size+j] = 0.0;
		}
	}

	double P[size];
	P[0] = 0.5;
	P[1] = 0.2;
	P[4] = 0.2;
	P[2] = 0.05;
	P[3] = 0.05;
	
	double sum = 0;
	for(int i = 0; i < size; i++)
		sum += P[i];

	for(int i = 0; i < size; i++)
		P[i] /= sum; 

	printf("target distribution:\n"); 
	for(int i = 0; i < size; i++)
		printf("P[%i] = %f\n", i, P[i]);

	double Pacc[size*size];
	for(int i = 0; i < size; i++){
		for(int j = 0; j < size; j++){
			Pacc[i*size+j] = minimum(1.0, P[j]/P[i]);
		}
	}

	printf("Pacc values:\n"); 
	for(int i = 0; i < size; i++){
		for(int j = 0; j < size; j++){
			printf("Pacc[%i] = %f,  ", i*size+j, Pacc[i*size+j]);
		}
		printf("\n");
	}
	printf("\n");


	for(int i = 0; i < size; i++){
		for(int j = 0; j < size; j++){

			if(i == j){
				T[i*size+j] = (1.0/5.0)*Pacc[i*size+j];
				for(int k = 0; k < size; k++){
					if( i != k ){
						T[i*size+j] += (1.0/5.0)*(1.0-Pacc[i*size+k]);
					}
				}
			}
			if(i != j)
				T[i*size+j] = (1.0/5.0)*Pacc[i*size+j];
		}
	}

	printf("T values:\n"); 
	for(int i = 0; i < size; i++){
		for(int j = 0; j < size; j++){
			printf("T[%i] = %f,  ", i*size+j, T[i*size+j]);
		}
		printf("\n");
	}
	printf("\n");


	double u[size];
	double v[size];

	for(int i = 0; i < size; i++)
		v[i] = 0;

	v[0] = 0.5;
	v[1] = 0.5;


	for(int k = 0; k < 64; k++){

		printf("iteration %i\n", k);

		mul(T,v,u);

		for(int i = 0; i < size; i++)
			printf("v[%i] = %f,  ", i, v[i]);
		printf("\n");

		for(int i = 0; i < size; i++)
			v[i] = u[i];
	}

	printf("target distribution:\n"); 
	for(int i = 0; i < size; i++)
		printf("P[%i] = %f,  ", i, P[i]);
	printf("\n");

	return 1;
}
