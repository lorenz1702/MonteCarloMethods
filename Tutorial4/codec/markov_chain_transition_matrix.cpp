#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <fstream>

#define size 5

int mul(double *T, double *v, double *u){

	int i,k;

	for(i = 0; i < size; i++){
	
		u[i] = 0;

		for(k = 0; k < size; k++){

			u[i] += T[i*size+k] * v[k];

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


	for(int i = 0; i < size; i++){
		for(int j = 0; j < size; j++){

			if(i == j)
				T[i*size+j] = 1.0/3.0;

			if(i == j + 1 && j + 1 < size)
				T[i*size+j] = 1.0/3.0;
			if(i == j - 1 && j - 1 >= 0)
				T[i*size+j] = 1.0/3.0;

				T[0*size+size-1] = 1.0/3.0;
				T[(size-1)*size+0] = 1.0/3.0;
		}
	}

/*
	T[0] = 1.0/3.0;
	T[1] = 1.0/3.0;
	T[2] = 0.0;
	T[3] = 0.0;
	T[4] = 1.0/3.0;

	T[5] = 1.0/3.0;
	T[6] = 1.0/3.0;
	T[7] = 1.0/3.0;
	T[8] = 0.0;
	T[9] = 0.0;

	T[10] = 0.0;
	T[11] = 1.0/3.0;
	T[12] = 1.0/3.0;
	T[13] = 1.0/3.0;
	T[14] = 0.0;

	T[15] = 0.0;
	T[16] = 0.0;
	T[17] = 1.0/3.0;
	T[18] = 1.0/3.0;
	T[19] = 1.0/3.0;

	T[20] = 1.0/3.0;
	T[21] = 0.0;
	T[22] = 0.0;
	T[23] = 1.0/3.0;
	T[24] = 1.0/3.0;
*/
	double u[size];
	double v[size];

	for(int i = 0; i < size; i++)
		v[i] = 0;

	v[0] = 1.0;

	double o;
	
	o = observable(v);

	printf("o = %e\n", o);
	printf("\n");


///////////////////////////////////////////////////////////////
//this part for writing the result
std::ofstream fileFFT("observable.csv");

// Check if the file is successfully opened
if (!fileFFT) {
	std::cerr << "Error opening file!" << std::endl;
	return 1;
}
// calculating the observable
	for(int k = 0; k < 200; k++){

		mul(T,v,u);

		o = observable(v);

		printf("o = %e\n", o);
		printf("\n");

		fileFFT <<k <<", "<< o <<"\n";
		for(int i = 0; i < size; i++)
			v[i] = u[i];
	}

fileFFT.close();

	return 1;

}
