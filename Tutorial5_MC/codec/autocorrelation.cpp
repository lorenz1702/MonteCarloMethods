#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define SIZE 200

int main(int argc, char* argv[]){

	int* array;

	array = (int*) malloc(10000000*sizeof(int));

	FILE *f;

	f = fopen(argv[1], "r+");

	int size = 0;
	int tmp;
	
	double obs_mean = 0;
	double obs_error = 0;

	while(!feof(f)){
		fscanf(f, "%i\n", &tmp);
		array[size] = tmp;
		obs_mean += fabs(tmp);
		obs_error += tmp*tmp;
		size++;
	}

	fclose(f);

	printf("##Read %i history entries\n", size);

	double autocorr[SIZE];
	double autocorre[SIZE];
	for(int tau = 0; tau < SIZE; tau++){
		autocorr[tau] = 0;
		autocorre[tau] = 0;
	}

	double tauint = 0;
	double tauint2 = 0;

	double mean = 0;
	for(int i = 0; i < size; i++){
		mean += array[i];
	}

	mean /= (1.0*size);

	int integrate = 1;
	int flag = 1;

	for(int tau = 0; tau < SIZE; tau++){
		printf("calculating for tau = %i\n", tau);
		for(int i = 0; i < size - tau; i++){
			autocorr[tau] += (array[i]-mean)*(array[i+tau]-mean);
			autocorre[tau] += ((array[i]-mean)*(array[i+tau]-mean))*((array[i]-mean)*(array[i+tau]-mean));
		}
		autocorre[tau] = sqrt(autocorre[tau]*(size-tau) - autocorr[tau]*autocorr[tau])/sqrt(1.0*(size-tau))/(1.0*(size-tau));
		autocorr[tau] /= (1.0*(size - tau));

		if(autocorr[tau] < 0 && flag == 1){
			integrate = tau;
			flag = 0;
		}
	}

	printf("integration limit = %i\n", integrate);

	for(int tau = 0; tau < SIZE; tau++){
		printf("tau %i autocorr(tau) = %f\n", tau, autocorr[tau]/autocorr[0]);
		if(tau < integrate){
			tauint += autocorr[tau]/autocorr[0];
			tauint2 += (1.0-1.0*tau/integrate)*autocorr[tau]/autocorr[0];

		}
	}

	printf("tauint = %e\n", tauint);
	printf("tauint2 = %e\n", tauint2);

//	for(int tau = 0; tau < SIZE; tau++){
//		printf("%i %e %e\n", tau, autocorr[tau]/autocorr[0], autocorre[tau]/autocorr[0]);
//		//printf("%i %e %e\n", tau, autocorr[tau], autocorre[tau]);
//
//	}

	double sigma_square = (obs_error*size - obs_mean*obs_mean)/(1.0*size)/(1.0*size);

	tauint = 0.5 + tauint; ///sigma_square;
	tauint2 = 0.5 + tauint2; ///sigma_square;

	printf("tauint = %e\n", tauint);
	printf("tauint2 = %e\n", tauint2);

	printf("mean naive_std std_with_autocorrelations\n");
	printf("%e \t %e \t %e\n", obs_mean/(1.0*size), sqrt(sigma_square)/sqrt(1.0*size), sqrt(sigma_square)/sqrt(1.0*size)*sqrt(2*tauint));
	printf("%e \t %e \t %e\n", obs_mean/(1.0*size), sqrt(sigma_square)/sqrt(1.0*size), sqrt(sigma_square)/sqrt(1.0*size)*sqrt(2*tauint2));

	return 1;
}

