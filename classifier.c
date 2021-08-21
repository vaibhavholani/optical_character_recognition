#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>      
#include <sys/types.h>  
#include <sys/wait.h>  
#include <string.h>
#include <math.h>
#include "knn.h"

/**
 * main() takes in the following command line arguments.
 *   -K <num>:  K value for kNN (default is 1)
 *   -d <distance metric>: a string for the distance function to use
 *          euclidean or cosine (or initial substring such as "eucl", or "cos")
 *   -p <num_procs>: The number of processes to use to test images
 *   -v : If this argument is provided, then print additional debugging information
 *        (You are welcome to add print statements that only print with the verbose
 *         option.  We will not be running tests with -v )
 *   training_data: A binary file containing training image / label data
 *   testing_data: A binary file containing testing image / label data
 *   (Note that the first three "option" arguments (-K <num>, -d <distance metric>,
 *   and -p <num_procs>) may appear in any order, but the two dataset files must
 *   be the last two arguments.
 * 
 *
 */
void usage(char *name) {
    fprintf(stderr, "Usage: %s -v -K <num> -d <distance metric> -p <num_procs> training_list testing_list\n", name);
}

int main(int argc, char *argv[]) {

    int opt;
    int K = 1;             // default value for K
    char *dist_metric = "euclidean"; // default distant metric
    int num_procs = 1;     // default number of children to create
    int verbose = 0;       // if verbose is 1, print extra debugging statements
    int total_correct = 0; // Number of correct predictions

    while((opt = getopt(argc, argv, "vK:d:p:")) != -1) {
        switch(opt) {
        case 'v':
            verbose = 1;
            break;
        case 'K':
            K = atoi(optarg);
            break;
        case 'd':
            dist_metric = optarg;
            break;
        case 'p':
            num_procs = atoi(optarg);
            break;
        default:
            usage(argv[0]);
            exit(1);
        }
    }

    if(optind >= argc) {
        fprintf(stderr, "Expecting training images file and test images file\n");
        exit(1);
    } 

    char *training_file = argv[optind];
    optind++;
    char *testing_file = argv[optind];
  
    // Set which distance function to use
    double (*fptr)(Image *, Image *);
    if (strncmp(dist_metric, "euclidean", strlen(dist_metric)) == 0){
        fptr = distance_euclidean;
    }
    else if (strncmp(dist_metric, "cosine", strlen(dist_metric)) == 0) {
        fptr = distance_cosine;
    }
    else {
        fprintf(stderr, "Usage for -d is euclidean or cosine ");
        exit(1);
    }

    // Load data sets
    if(verbose) {

        fprintf(stderr,"- Loading datasets...\n");
    }
    
    Dataset *training = load_dataset(training_file);
    if ( training == NULL ) {
        fprintf(stderr, "The data set in %s could not be loaded\n", training_file);
        exit(1);
    }

    Dataset *testing = load_dataset(testing_file);
    if ( testing == NULL ) {
        fprintf(stderr, "The data set in %s could not be loaded\n", testing_file);
        exit(1);
    }

    // Create the pipes and child processes who will then call child_handler
    if(verbose) {
        printf("- Creating children ...\n");
    }


    // Finding N
    int test_set_size = testing->num_items;
    int start_idx = 0;
    int N;

    // An array to keep track of all the fd_csend pipes the parent has to read 
    int csend_array[(2*num_procs)];


    for (int i = 0; i < num_procs; i++){

        // Choosing the value of N
        
        if (i < (test_set_size%num_procs)){
            N = (int) ceil((double)test_set_size / num_procs);
        }
        else {
            N = (int) floor((double)test_set_size / num_procs);
        }
        // The pipe used to send data from the parent to the child
        int fd_psend[2];
        // The pipe used to send data from the child to the parent
        int* fd_csend = &(csend_array[(2*i)]);

        // Making pipes 
        if (pipe(fd_psend) == -1) {
            perror("pipe");
            exit(1);
        };

        if (pipe(fd_csend) == -1) {
            perror("pipe");
            exit(1);
        };

        // making a child
        int id = fork();
        if (id == 0){
            if (close(fd_psend[1]) < 0) {
                perror("close");
                exit(1);
            }
            if (close(fd_csend[0]) < 0) {
                perror("close");
                exit(1);
            }
            child_handler(training, testing, K, fptr, fd_psend[0], fd_csend[1]);
            free_dataset(training);
            free_dataset(testing);
            exit(0);
        }
        else if (id > 0) {
            // As the parent will write using psend, close psend read
            if (close(fd_psend[0]) < 0 ){
                perror("close");
                exit(1);
            }
            // Similarly, no writing to csend
            if (close(fd_csend[1]) < 0 ){
                perror("close");
                exit(1);
            }

            int check = 1;
            while (check) {
                // writing to the pipe
                int value = write(fd_psend[1],&start_idx, sizeof(int));
                if (value == sizeof(int)){
                    check = 0;
                }
                else if (value == -1){
                    perror("write");
                    exit(1);
                }
            }

            check = 1;

            while (check) {
                // writing to the pipe
                int value = write(fd_psend[1],&N, sizeof(int));
                if (value == sizeof(int)){
                    check = 0;
                }
                else if (value == -1){
                    perror("write");
                    exit(1);
                }
            }

            // incrementing start_idx
            start_idx = start_idx + N;

            // Parent finished writing to child
            if (close(fd_psend[1]) < 0 ){
                perror("close");
                exit(1);
            }
        }
        else {
            perror("fork");
            exit(1);
        }
    }

    // Distribute the work to the children by writing their starting index and
    // the number of test images to process to their write pipe

    if(verbose) {
        printf("- Waiting for children...\n");
    }
    // When the children have finised, read their results from their pipe
    for (int i = 0; i < num_procs; i++){

        int curr_fd = csend_array[i*2];
        int num_correct;
        int check = 1;
            while (check) {
                // reading from the pipe
                int value = read(curr_fd, &num_correct, sizeof(int));
                if (value == sizeof(int)){
                    check = 0;
                    total_correct = total_correct + num_correct;
                }
                else if (value == -1){
                    perror("read");
                    exit(1);
                }
            }
        
    }

    // closing all the open csend file discriptors
    for (int i = 0; i < num_procs; i++){
        int curr_fd = csend_array[i*2];

        if (close(curr_fd) < 0 ){
                perror("close");
                exit(1);
            }
    }
    

    // Wait for children to finish
 
    for (int i = 0; i < num_procs; i++) {
        int status;
        if (wait(&status) < 0)  {
            perror("wait");
            exit(1);
        
        }

        if (WIFEXITED(status)) {
            if (WEXITSTATUS(status) == 1) {
                fprintf(stderr, "Problem with reading or writing in children processes");
                exit(1);
            }
        }
    }

    if(verbose) {
        printf("Number of correct predictions: %d\n", total_correct);
    }

    // This is the only print statement that can occur outside the verbose check
    printf("%d\n", total_correct);

    // Clean up any memory, open files, or open pipes
    free_dataset(training);
    free_dataset(testing);

    return 0;
}
