#include <stdio.h>
#include <stdlib.h>
#include <unistd.h> 
#include <string.h>
#include "knn.h"


int main(int argc, char **argv) {
    if(argc != 2) {
        fprintf(stderr, "Usage: %s filename\n", argv[0]);
        exit(1);
    }

    Dataset *data = load_dataset(argv[1]);

    // Compute the distance between the first two images in the data set

    double cos_distance = distance_cosine(&data->images[0], &data->images[1]);
    double euc_distance = distance_euclidean(&data->images[0], &data->images[1]);

    printf("Cosine distance = %f\n", cos_distance);
    printf("Euclidean distance = %f\n", euc_distance);
    return 0;
}
