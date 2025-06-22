#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define SIZE 2000

int main(int argc, char* argv[]){

	int* array;

	array = (int*) malloc(10000000*sizeof(int));

	FILE *f;

	f = fopen(argv[1], "r+");

	int size = 0;
	float tmp;
	
	double obs_mean = 0;
	double obs_error = 0;

	while(!feof(f)){
		fscanf(f, "%f\n", &tmp);
		array[size] = tmp;
		obs_mean += fabs(tmp);
		obs_error += tmp*tmp;
		size++;
	}

	fclose(f);

	double sigma_square = (obs_error*size - obs_mean*obs_mean)/(1.0*size)/(1.0*size);

	printf("%i %e \t %e\n", 1, obs_mean/(1.0*size), sqrt(sigma_square)/sqrt(1.0*size));

	for(int i = 1; i < 70000; i*=2){ //i is the size of each chunk

		int stat = 0;

		obs_mean = 0;
		obs_error = 0;

		int count = 0;
		double obs_mean_tmp = 0;

		for(int t = 0; t < size; t++){

			if(count == i){
				count = 0;
				obs_mean += (obs_mean_tmp/(1.0*i));
				obs_error += (obs_mean_tmp/(1.0*i))*(obs_mean_tmp/(1.0*i));
				obs_mean_tmp = 0;
				stat++;
			}

			if(count < i){
				obs_mean_tmp += array[t];
				count++;
			}
		}

		double sigma_square = (obs_error*stat - obs_mean*obs_mean)/(1.0*stat)/(1.0*stat);

		printf("%i %e \t %e\n", i, obs_mean/(1.0*stat), sqrt(sigma_square)/sqrt(1.0*stat));

	}

	return 1;
}

