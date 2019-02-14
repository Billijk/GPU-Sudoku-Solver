#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "methods.h"

int* read_board(const char* file_name, int* N) {
    FILE* fp = fopen(file_name, "r");
    int* board = NULL;

    fscanf(fp, "%d", N);
    int total = *N * *N;
    board = calloc(total, sizeof(int));
    int i;
    for (i = 0; i < total; ++ i)
        fscanf(fp, "%d", board + i);
    return board;
}


int main(int argc, char* argv[]) {

    if (argc < 4) {
        printf("Usage:\n");
        printf("./main <input> <output> <method>\n");
        exit(-1);
    }

    int N;
    int* board = read_board(argv[1], &N);
    int n = sqrt((double)N);
    printf("Start to solve a %d x %d Sudoku.\n", N, N);

    FILE* fp = fopen(argv[2], "w");
    int method = 1;
    if (argc == 4) method = (int) atoi(argv[3]);

    int ans;
    switch (method) {
        case 0:
            ans = backtracking(board, n, fp);
            break;
        case 1:
            ans = dancing_links(board, n, fp);
            break;
        default:
            printf("Unrecognized method. Now only the following methods are supported: \n");
            printf("0 - backtracking \n");
            printf("1 - DLX (default)\n");
            break;
    }
        
    if (ans) printf("An answer is found and saved to %s.\n", argv[2]);
    else printf("No answer is found.\n");

    free(board);
    return 0;
}