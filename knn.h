#pragma once


#define WIDTH 28
#define NUM_PIXELS WIDTH * WIDTH

/* This struct stores the data for an image */
typedef struct {
    int sx;               // x resolution
    int sy;               // y resolution
    unsigned char *data;  // List of `sx * sy` pixel color values [0-255]
} Image;

/* This struct stores the images / labels in the dataset */
typedef struct {
    int num_items;          // Number of images in the dataset
    Image *images;          // List of `num_items` Image structs
    unsigned char *labels;  // List of `num_items` labels [0-9]
} Dataset;

double distance_euclidean(Image *a, Image *b);

Dataset *load_dataset(const char *filename);
void free_dataset(Dataset *data);

double distance_cosine(Image *a, Image *b);
int knn_predict(Dataset *data, Image *img, int K, double (*fptr)(Image *,Image *));
void child_handler(Dataset *training, Dataset *testing, int K, double (*fptr)(Image *, Image *),int p_in, int p_out);
