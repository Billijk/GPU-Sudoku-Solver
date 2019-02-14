#include <stdio.h>

#ifndef METHODS_H
#define METHODS_H

#define id(i, j, N) ((i) * N + (j))
#define ROW(p, N) ((p) / N)
#define COL(p, N) ((p) % N)
#define BOX(p, n) ((p) / (n * n * n) * n + ((p) % (n * n)) / n)

// check if a solution is valid on a NxN field (N = n^2)
int check_board(int* board, int n);
// print a board to file
void print_board(int* board, int N, FILE* fp);
// check if num at position p is valid given a partially filled board
int check_partial_board(int* board, int n, int p, int num);
// find candidates for each non-filled location
void initial_check(int* board, int n, int** valid_candidates);


// Brute Force Backtracking
int backtracking(int *board, int n, FILE* fp);

// DLX
#define UNLOAD(dlx, dlx_props, dlx_size) \
    up    = (dlx); \
    down  = (dlx) + dlx_size; \
    left  = (dlx) + dlx_size * 2; \
    right = (dlx) + dlx_size * 3; \
    col   = (dlx_props); \
    row   = (dlx_props) + dlx_size;
void remove_column(int id, int *dlx, int *dlx_props, int dlx_size);
void restore_column(int id, int *dlx, int *dlx_props, int dlx_size);

int build_dancing_links(int* col_ids, int* row_ids, int n, int **dlx_ptr);
int convert_matrix(int *board, int **valid_candidates, int n, int **cols_ptr, int **rows_ptr, int** convert_table_ptr);
void convert_answer_print(int *ans, int *convert_table, int N, FILE *fp);

int dancing_links(int *board, int n, FILE* fp);

#endif // METHODS_H
