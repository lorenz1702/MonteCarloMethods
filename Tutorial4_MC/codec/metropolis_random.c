#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "ranlux-3.4/ranlux.h"


#define size 5

double minimum(double a, double b){
	if( a > b )
		return b;
	else
		return a;
}

double observable(double *v){

	double o = 0;

	for(int i = 0; i < size; i++){
		o += v[i]*i;
	}

	return o;

}

int main(int argc, char* argv[]){

	rlxs_init(2,(unsigned long int)(atoi(argv[1])));

	double P[size];
	P[0] = 0.5;
	P[1] = 0.2;
	P[4] = 0.2;
	P[2] = 0.05;
	P[3] = 0.05;
	
	//theoretical expectation:
	//0.2*1+0.2*4+0.05*2+0.05*3=0.2+0.8+0.1+0.15=1.25

	double sum = 0;
	for(int i = 0; i < size; i++)
		sum += P[i];

	printf("sum = %e\n", sum);

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


	int s;

	float ini[1];
        ranlxs(ini,1);

	if(ini[0] < 0.2)
		s = 0;
	if(ini[0] >= 0.2 && ini[0] < 0.4)
		s = 1;
	if(ini[0] >= 0.4 && ini[0] < 0.6)
		s = 2;
	if(ini[0] >= 0.6 && ini[0] < 0.8)
		s = 3;
	if(ini[0] >= 0.8)
		s = 4;
	printf("initial state: s = %i\n", s);

	int acc = 0;
	double o = 0;
	double oe = 0;

	for(int iter = 0; iter < 10000000; iter++){

		float p[1];
        	ranlxs(p,1);
		int sp = 0;

		if(p[0] < 0.2)
			sp = 0;
		if(p[0] >= 0.2 && p[0] < 0.4)
			sp = 1;
		if(p[0] >= 0.4 && p[0] < 0.6)
			sp = 2;
		if(p[0] >= 0.6 && p[0] < 0.8)
			sp = 3;
		if(p[0] >= 0.8)
			sp = 4;

		//accept/reject
			ranlxs(p,1);

			if( p[0] < Pacc[s*size+sp] ){

				acc++;

			s = sp;

		}

		o += 1.0*s;
		oe += 1.0*s*s;

		if(iter%100000 == 0){

			printf("acceptance = %f\n", 1.0*acc/(iter+1));
			printf("observable exp = %f %f\n", 1.0*o/(100000.0), sqrt(oe*100000.0-o*o)/100000.0/sqrt(100000.0));

			o = 0.0;
			oe = 0.0;
		}
        }

	return 1;
}
