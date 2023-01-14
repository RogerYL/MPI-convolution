#include<stdio.h>
#include<stdlib.h>
#define __USE_C99_MATH
#include<string.h>
#include<stdbool.h>
#include<math.h>
#include<mpi.h>
#include <omp.h>
#include<sys/time.h>

long int product(int *array, int n) {
    long int product = 1;
    for(int i=0; i<n; i++) {
        product *= array[i];
    }
    return product;
}

int *read_dims(char *filename) {
    FILE *file = fopen(filename,"r");
    
    if(file == NULL) {
        printf("Unable to open file: %s", filename);
        return NULL;
    }

    char firstline[500];
    fgets(firstline, 500, file);
    
    int line_length = strlen(firstline);

    int num_dims = 0;
    for(int i=0; i<line_length; i++) {
        if(firstline[i] == ' ') {
            num_dims++;
        }
    }
    
    int *dims = malloc((num_dims+1)*sizeof(int));
    dims[0] = num_dims;
    const char s[2] = " ";
    char *token;
    token = strtok(firstline, s);
    int i = 0;
    while( token != NULL ) {
        dims[i+1] = atoi(token);
        i++;
        token = strtok(NULL, s);
    }
    fclose(file);
    return dims;
}

float * read_array(char *filename, int *dims, int num_dims) {
    FILE *file = fopen(filename,"r");

    if(file == NULL) {
        printf("Unable to open file: %s", filename);
        return NULL;
    }

    char firstline[500];
    fgets(firstline, 500, file);

    //Ignore first line and move on since first line contains 
    //header information and we already have that. 

    long int total_elements = product(dims, num_dims);

    float *one_d = malloc(sizeof(float) * total_elements);
    for(int i=0; i<total_elements; i++) {
        fscanf(file, "%f", &one_d[i]);
    }
    fclose(file);
    return one_d;
}

int write_array(char *filename, int batch, int row, int col, float *output){
    int size = batch * row * col;
    FILE *file = fopen(filename,"w");

    if(file == NULL) {
        printf("Unable to open file: %s", filename);
        return -1;
    }

    if (file != NULL) {
        fprintf(file, "%d ", batch);
        fprintf(file, "%d ", row);
        fprintf(file, "%d ", col);
        fprintf(file, "\n");
    }

    for(int i=0; i<size; i++) {
        fprintf(file, "%.6f ", output[i]);
    }

    fclose(file);
    return 1;
}

int main(int argc, char *argv[]) {
    //Define the number and id of threads
    int process_num, process_id;
    //Packets for broadcasting
    int mat_p[8];
    //Counting the units to be operated on each level
    int counter = 0;
    //Time Record
    double MPI_timer[4];
    //Raw data read from thread 0, belongs to address header, no space allocated
    float *input_master = NULL;
    float *kernel_master = NULL;
    float *check_master = NULL;
    //The final output to the space, which is now still the address header
    float *output = NULL;
    //Definition of packets used by gather
    float *recv_block = NULL;
    int compareOutput = 1;
    bool match = true;
    char input_filename[500];
    char kernel_filename[500];
    char check_filename[500];
    MPI_Init(&argc, &argv);
    MPI_timer[0] = MPI_Wtime();
    MPI_Comm_rank(MPI_COMM_WORLD, &process_id);
    MPI_Comm_size(MPI_COMM_WORLD, &process_num);
    if(process_id == 0){
        if(argc != 4) {
            printf("Usage: %s <filename_input> <filename_kernel> <filename_expected_output>\n", argv[0]);
            return -1;
        }
        strcpy(input_filename, argv[1]);
        strcpy(kernel_filename, argv[2]);
        strcpy(check_filename, argv[3]);
        int *input_dims_original = read_dims(input_filename);
        if(input_dims_original == NULL) {
            return -1;
        }
        int input_num_dims = input_dims_original[0];
        int *input_dims = input_dims_original+1;
        input_master = read_array(input_filename, input_dims, input_num_dims);
        if(input_master == NULL) {
            return -1;
        }
        int *kernel_dims_original = read_dims(kernel_filename);
        if(kernel_dims_original == NULL) {
            return -1;
        }
        int kernel_num_dims = kernel_dims_original[0];
        int *kernel_dims = kernel_dims_original+1;
        kernel_master = read_array(kernel_filename, kernel_dims, kernel_num_dims);
        if(kernel_master == NULL) {
            return -1;
        }
        int num_row = input_dims[1]-kernel_dims[0]+1;//matrix row(hight) need to be calculated
	    int num_col = input_dims[2]-kernel_dims[0]+1;//matrix col(width) need to be calculated
        //Effective calculation unit
        int counter = num_row * num_col;
        printf("calculation unit %d, have threads %d\n", counter, process_num);
        int calculate_center = (int)((kernel_dims[0]-1)/2);
        //Place all parameters to be used in the broadcast package
        mat_p[0] = input_dims[0];
        mat_p[1] = input_dims[1];
        mat_p[2] = input_dims[2];
        mat_p[3] = kernel_dims[0];
        mat_p[4] = calculate_center;
        mat_p[5] = num_col;
        mat_p[6] = num_row;
        mat_p[7] = counter;
        int size = mat_p[0] * mat_p[1] * mat_p[2];
        //Creat memory for output in process 0, when only process 0 can use output
        output = malloc(sizeof(float) * size);
        //Initialize output
        for(int n = 0; n < size; n++){
            *(output + n) = *(input_master + n);
        }
    }
    MPI_timer[1] = MPI_Wtime();
    //Broadcast parameters to all threads
    MPI_Bcast(&mat_p, 8, MPI_INT, 0, MPI_COMM_WORLD);
    //Calculate all parameters coming out of process 0
    int layer = mat_p[1] * mat_p[2];
    int size_input = mat_p[0] * mat_p[1] * mat_p[2];
    int size_kernel = mat_p[3] * mat_p[3];
    int size_calculation = mat_p[5] * mat_p[6];
    int array_length = mat_p[0] * size_calculation;
    //Create memory space for input and kernel storage for each process
    float *input = malloc(sizeof(float) * size_input);
    float *kernel = malloc(sizeof(float) * size_kernel);
    //Create space for data to be placed after each thread's calculation
    float *calculation_block = malloc(sizeof(float) * array_length);
    if(process_id == 0){
        //Initialize in process 0, input and kernel in all processes
        input = input_master;
        kernel = kernel_master;
    }
    //Broadcast input and kernel to all processes
    MPI_Bcast(input, size_input, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(kernel, size_kernel, MPI_FLOAT, 0, MPI_COMM_WORLD);
    //MPI_Bcast(check, size_input, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(calculation_block, array_length, MPI_FLOAT, 0, MPI_COMM_WORLD);
    if(process_id == 0){
        //Initialize the package space for gather
        int all_array = process_num * array_length;
        recv_block = malloc(sizeof(float) * all_array);
    }

    //coordinate[0] = start_row, coordinate[1] = start_col, coordinate[2] = elements
    //coordinate[3] = flag, if this threads have task flag = 1, if not flag = 0;
    int *coordinate = malloc(sizeof(int) * 4);
    if(process_id >= mat_p[7]){
        *(coordinate + 0) = 0;
        *(coordinate + 1) = 0;
        *(coordinate + 2) = 0;
        *(coordinate + 3) = 0;
        //MPI_Send(coordinate, 4, MPI_INT, process_id, 99, MPI_COMM_WORLD);
    }else{
        //Calculate the start address of each thread and the number of data to be calculated
        int max_np;
        if(mat_p[7] < process_num){
            max_np = mat_p[7];
        }else{
            max_np = process_num;
        }
        int elements = (int)(mat_p[7] / max_np);
        int rest = mat_p[7] - (elements * max_np);
        int already_sent = process_id * elements;
        int start_row = (int)(already_sent / mat_p[5]);
        int start_col = already_sent - (start_row * mat_p[5]);
        
        *(coordinate + 0) = start_row;
        *(coordinate + 1) = start_col;
        if (process_id == max_np - 1)
        {
            *(coordinate + 2) = elements + rest;
        }else{
            *(coordinate + 2) = elements;
        }
        *(coordinate + 3) = 1;
    }

    //Threads start computing
    int row_recv = *(coordinate + 0);
    int col_recv = *(coordinate + 1);
    int elements = *(coordinate + 2);
    int flag = *(coordinate + 3);
    if(flag == 1){
        //Effective threads for work
        for(int e = 0; e < elements; e++){
            //Arithmetic from the start address
            int relocate = row_recv * mat_p[5] + col_recv + e;
            int new_row = (int)(relocate / mat_p[5]);
            int new_col = relocate - (new_row * mat_p[5]);
            for(int batch = 0; batch < mat_p[0]; batch++){
                float temp = 0;
                float each = 0;
                for(int row = 0; row < mat_p[3]; row++){
                    for(int col = 0; col < mat_p[3]; col++){
                        each = *(input + batch * layer + (new_row + row) * mat_p[2] + new_col + col) * *(kernel + col + row * mat_p[3]);
                        temp = temp + each;
                    }
                }
                *(calculation_block + e * mat_p[0] + batch) = temp;
            }
        }
    }else{
        //Non-valid threads for work
        printf("Rank %d is not working \n", process_id);
    }
    //gather the results of each thread's operations
    MPI_Gather(calculation_block, array_length, MPI_FLOAT, recv_block, array_length, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_timer[2] = MPI_Wtime();
    printf("%f\n", MPI_timer[2] - MPI_timer[1]);
    if(process_id == 0){
        //Final redistribution of data at process 0
        float *after_cal = malloc(sizeof(float) * array_length);
        int counting = 0;
        int working_np;
        if(mat_p[7] < process_num){
            working_np = mat_p[7];
        }else{
            working_np = process_num;
        }
        int elements = (int)(mat_p[7] / working_np);
        int rest = mat_p[7] - (elements * working_np);
        for(int t = 0; t < working_np - 1; t++){
            for(int e = 0; e < elements; e++){
                for(int b = 0; b < mat_p[0]; b++){
                    *(after_cal + counting) = *(recv_block + t * array_length + e * mat_p[0] + b);
                    counting++;
                }
            }
        }
        for(int e = 0; e < elements + rest; e++){
            for(int b = 0; b < mat_p[0]; b++){
                *(after_cal + counting) = *(recv_block + (working_np - 1) * array_length + e * mat_p[0] + b);
                counting++;
            }
        }
        for(int b = 0; b < mat_p[0]; b++){
            for(int row = 0; row < mat_p[6]; row++){
                for(int col = 0; col < mat_p[5]; col++){
                    *(output + b * layer + (row + mat_p[4]) * mat_p[2] + col + mat_p[4]) = *(after_cal + mat_p[0] * (row * mat_p[5] + col) + b);
                }
            }
        }
    }
    MPI_timer[3] = MPI_Wtime();
    printf("-------------------------------------%f\n", MPI_timer[3] - MPI_timer[1]);
    if(process_id == 0){
        //Process 0 does the file writing
        //printf("-----------------------finish calculation-----------------------\n");
        int write = write_array(check_filename, mat_p[0], mat_p[1], mat_p[2], output);
        if(write == 1) {
            printf("Writing successful!\n");
        }
    }
    MPI_Finalize();
    return !match;
}
