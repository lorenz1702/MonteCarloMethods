#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "ranlux-3.4/ranlux.h"

#define N 5

int main(void){

	rlxs_init(2,(int)(3));

	int state = 1;
	int nfre[5];
	for (int k = 0; k < N; k++)
	{
		nfre[k] = 0;
	}

	for(int i = 0; i < 10000; i++){

		//one step:
		float p[1];
		
	        ranlxs(p,1);

		if( p[0] < 1.0/3.0 )
			state = (state+1)%N;
		else if( p[0] >= 1.0/3.0 && p[0] < 2.0/3.0 )
			state = state;
		else
			state = (state-1+N)%N;

		// this part to store occurence of state
		for (int k = 0; k < N; k++)
		{
			if (state==k){
				nfre[k] +=1;
			}
		}
		
		printf("iteration %i: state %i\n", i, state);

	}

	
// this part to store occurence of state
printf("--------------------------------\n");
for (int k = 0; k < N; k++)
{
	printf("state %i:  occurence %i\n", k, nfre[k]);

}
// ///////////////////////////////////////////////////////////////
// //this part for writing the result
// std::ofstream fileFFT("probability.csv");

// // Check if the file is successfully opened
// if (!fileFFT) {
// 	std::cerr << "Error opening file!" << std::endl;
// 	return 1;
// }

// fileFFT.close();
	return 1;
}

