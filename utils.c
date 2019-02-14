#include "methods.h"

// check if a solution is valid on a NxN field (N = n^2)
// Using the algorithm from
// https://stackoverflow.com/questions/289537/a-cool-algorithm-to-check-a-sudoku-field
int check_board(int* board, int n) {
    int N = n * n, i, j, k;
    int true_sum = (1 + N) * N / 2;
    // check sum on each row -- O(N^2)
    for (i = 0; i < N; ++ i) {
        int sum = 0;
        for (j = 0; j < N; ++ j)
            sum += board[id(i, j, N)];
        if (sum != true_sum) return 0;
    }
    
    // check sum on each column -- O(N^2)
    for (i = 0; i < N; ++ i) {
        int sum = 0;
        for (j = 0; j < N; ++ j)
            sum += board[id(j, i, N)];
        if (sum != true_sum) return 0;
    }

    // check sum on each box -- O(N^2)
    for (i = 0; i < N; ++ i) {
        int base = (N * (i / n) + (i % n)) * n;
        int sum = 0;
        for (j = 0; j < N; ++ j) {
            int offset = N * (j / n) + (j % n);
            sum += board[base + offset];
        }
        if (sum != true_sum) return 0;
    }

    // check for duplicate numbers on each row -- O(N^3)
    for (i = 0; i < N; ++ i)
        for ( j = 0; j < N - 1; ++ j)
            for ( k = j + 1; k < N; ++ k)
                if (board[id(i, j, N)] == board[id(i, k, N)])
                    return 0;

    // check for duplicate numbers on each column -- O(N^3)
    for ( i = 0; i < N; ++ i)
        for ( j = 0; j < N - 1; ++ j)
            for ( k = j + 1; k < N; ++ k)
                if (board[id(j, i, N)] == board[id(k, i, N)])
                    return 0;

    // check for duplicate numbers on each box -- O(N^3)
    for ( i = 0; i < N; ++ i) {
        int base = (N * (i / n) + (i % n)) * n;
        int sum = 0;
        for ( j = 0; j < N - 1; ++ j) {
            int os1 = N * (j / n) + (j % n);
            for ( k = j + 1; k < N; ++ k) {
                int os2 = N * (k / n) + (k % n);
                if (board[base + os1] == board[base + os2])
                    return 0;
            }
        }
    }
    
    return 1;
}

// print a board to file
void print_board(int* board, int N, FILE* fp) {
    int i, j;
    for (i = 0; i < N; ++ i) {
        for (j = 0; j < N; ++ j) {
            fprintf(fp, "%d ", board[id(i, j, N)]);
        }
        fprintf(fp, "\n");
    }
    fprintf(fp, "\n");
}

// check if num at position p is valid given a partially filled board
int check_partial_board(int* board, int n, int p, int num) {
    int j;
    int N = n * n;
    int box_row = p / (n * N);
    int box_col = (p % N) / n;
    int box_top_left = box_row * n * N + box_col * n;
    int now_row = ROW(p, N);
    for (j = now_row * N; j < (now_row + 1) * N; ++ j)
        if (board[j] == num)
            return 0;
    // check col
    for (j = COL(p, N); j < N * N; j += N)
        if (board[j] == num)
            return 0;
    // check box
    for (j = 0; j < N; ++ j)
        if (board[box_top_left + (j / n) * N + (j % n)] == num)
            return 0;
    return 1;
}

// find candidates for each non-filled location
void initial_check(int* board, int n, int** valid_candidates) {
    int N = n * n, i, cand;
    for (i = 0; i < N * N; ++ i) {
        if (board[i] != 0) continue;
        for (cand = 0; cand < N; ++ cand) {
            if (check_partial_board(board, n, i, cand + 1)) 
                valid_candidates[i][cand] = 1;
        }
    }
}