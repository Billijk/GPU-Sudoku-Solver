#include "methods.h"
#include <stdlib.h>

// ===================================================
//  Method 1: brute force backtracking
// ===================================================
int backtracking_dfs(int *board, int n, int p, FILE* fp) {
    int N = n * n;
    int i;
    if (p == N * N) {
        // find a new solution, write to file
        print_board(board, N, fp);
        return 1;
    } else {
        if (board[p] == 0) {
            for (i = 0; i < N; ++ i) {
                if (check_partial_board(board, n, p, i + 1)) {
                    board[p] = i + 1;
                    int ret = backtracking_dfs(board, n, p + 1, fp);
                    if (ret) return ret;
                }
            }
            board[p] = 0;
        } else {
            return backtracking_dfs(board, n, p + 1, fp);
        }
        return 0;
    }
}

int backtracking(int *board, int n, FILE* fp) {
    int ans_num = backtracking_dfs(board, n, 0, fp);
    return ans_num;
}

// ==============================================
//  Method 2: Dancing Links X
// ==============================================

// remove the column indicator from the dancing links
// also remove all rows which attach to this column
void remove_column(int id, int *dlx, int *dlx_props, int dlx_size) {
    int *up, *down, *left, *right, *col, *row;
    UNLOAD(dlx, dlx_props, dlx_size);
    
    // first detach the column indicator
    right[left[id]] = right[id];
    left[right[id]] = left[id];

    // find every row of this column
    int row_id, elem;
    for (row_id = down[id]; row_id != id; row_id = down[row_id]) {
        // find every element in that row
        for (elem = right[row_id]; elem != row_id; elem = right[elem]) {
            // detach that element
            down[up[elem]] = down[elem];
            up[down[elem]] = up[elem];
        }
    }
}

// reverse step of remove_column
void restore_column(int id, int *dlx, int *dlx_props, int dlx_size) {
    int *up, *down, *left, *right, *col, *row;
    UNLOAD(dlx, dlx_props, dlx_size);

    // first detach the column indicator
    right[left[id]] = id;
    left[right[id]] = id;

    // find every row of this column
    int row_id, elem;
    for (row_id = down[id]; row_id != id; row_id = down[row_id]) {
        // find every element in that row
        for (elem = right[row_id]; elem != row_id; elem = right[elem]) {
            // attach that element
            down[up[elem]] = elem;
            up[down[elem]] = elem;
        }
    }
}

// solve the exact cover problem using DLX algorithm
int exact_cover(int *dlx, int *dlx_props, int dlx_size, int *ans, int *anscnt, int p) {

    int *up, *down, *left, *right, *col, *row;
    UNLOAD(dlx, dlx_props, dlx_size);
    
    if (right[0] == 0) {
        // every element has been covered
        *anscnt = p;
        return 1;
    }
    
    int current_col = right[0];
    int row_id = down[current_col];
    if (row_id == current_col)
        // this column has not been covered
        return 0;
    
    int elem;
    remove_column(current_col, dlx, dlx_props, dlx_size);
    for (;row_id != current_col; row_id = down[row_id]) {
        for (elem = right[row_id]; elem != row_id; elem = right[elem])
            remove_column(col[elem], dlx, dlx_props, dlx_size);
        ans[p] = row[row_id];
        int ret = exact_cover(dlx, dlx_props, dlx_size, ans, anscnt, p + 1);
        if (ret) return ret;
        for (elem = right[row_id]; elem != row_id; elem = right[elem])
            restore_column(col[elem], dlx, dlx_props, dlx_size);
    }
    restore_column(current_col, dlx, dlx_props, dlx_size);
    return 0;
}

int build_dancing_links(int* col_ids, int* row_ids, int n, int **dlx_ptr) {

    // calculate number of nodes
    int num_cols = 0, num_rows = 0, i;
    for (i = 0; i < n; ++ i) {
        if (col_ids[i] > num_cols)
            num_cols = col_ids[i];
        if (row_ids[i] > num_rows)
            num_rows = row_ids[i];
    }
    num_cols += 1;
    num_rows += 1;
    int count = num_cols + n + 1;

    // allocate memory for dancing links
    *dlx_ptr = calloc(count * 6, sizeof(int));
    int *up, *down, *left, *right, *col, *row;
    UNLOAD(*dlx_ptr, *dlx_ptr + 4 * count, count);
    
    // build column indicators first
    int now_id = 1;
    for (i = 0; i < num_cols; ++ i) {
        left[now_id] = now_id - 1;
        right[now_id - 1] = now_id;
        up[now_id] = down[now_id] = col[now_id] = now_id;
        now_id ++;
    }
    right[now_id - 1] = 0;
    left[0] = now_id - 1;

    // save pointers to one element in that row
    // for faster allocation of rows
    int* row_ptrs = calloc(num_rows, sizeof(int));

    // insert elements
    for (i = 0; i < n; ++ i, ++ now_id) {
        // add vertical edges
        int col_ptr_id = col_ids[i] + 1;
        col[now_id] = col_ptr_id;
        down[now_id] = down[col_ptr_id];
        down[col_ptr_id] = now_id;
        up[now_id] = col_ptr_id;
        up[down[now_id]] = now_id;

        // add horizontal edges
        int row_num = row[now_id] = row_ids[i];
        if (row_ptrs[row_num] == 0) {
            // first element in this row
            left[now_id] = right[now_id] = now_id;
            row_ptrs[row_num] = now_id;
        } else {
            int row_ptr_id = row_ptrs[row_num];
            right[now_id] = right[row_ptr_id];
            right[row_ptr_id] = now_id;
            left[now_id] = row_ptr_id;
            left[right[now_id]] = now_id;
        }
    }

    free(row_ptrs);
    return count;
}

// convert a Sudoku matrix to exact cover matrix
int convert_matrix(int *board, int **valid_candidates, int n, int **cols_ptr, int **rows_ptr, int** convert_table_ptr) {
    //https://www.jianshu.com/p/93b52c37cc65
    int N = n * n, i, j;
    int total = N * N;
    // first compute the number of 1
    int items_cnt = 0;
    for (i = 0; i < N * N; ++ i) {
        if (board[i] == 0) items_cnt += 4 * N;
        else items_cnt += 4;
    }

    *cols_ptr = (int*) malloc(sizeof(int) * items_cnt);
    *rows_ptr = (int*) malloc(sizeof(int) * items_cnt);
    *convert_table_ptr = (int*) malloc(sizeof(int) * items_cnt / 4);
    int *cols = *cols_ptr;
    int *rows = *rows_ptr;
    int *convert_table = *convert_table_ptr;

#define insert_number_into_matrix(num) { \
    /* a number is put into grid i */ \
    rows[elem] = row_num; \
    cols[elem ++] = i; \
    /* number num is put into row ROW(i) */ \
    rows[elem] = row_num; \
    cols[elem ++] = total + ROW(i, N) * N + num - 1;\
    /* number num is put into column COL(i) */ \
    rows[elem] = row_num; \
    cols[elem ++] = total * 2 + COL(i, N) * N + num - 1;\
    /* number num is put into box BOX(i) */ \
    rows[elem] = row_num; \
    cols[elem ++] = total * 3 + BOX(i, n) * N + num - 1;\
}

    int row_num = 0, elem = 0;
    for (i = 0; i < N * N; ++ i) {
        if (board[i] == 0) {
            for (j = 0; j < N; ++ j) {
                if (valid_candidates[i][j]) {
                    insert_number_into_matrix(j + 1);
                    convert_table[row_num] = (i * N) + j;
                    row_num ++;
                }
            }
        } else {
            insert_number_into_matrix(board[i]);
            convert_table[row_num] = (i * N) + board[i] - 1;
            row_num ++;
        }
    }

    return elem;
}

// convert an exact cover answer to a Sudoku answer and print
void convert_answer_print(int *ans, int *convert_table, int N, FILE *fp) {
    int i;
    int *answer_board = calloc(N * N, sizeof(int));
    for (i = 0; i < N * N; ++ i) {
        int pos_and_num = convert_table[ans[i]];
        answer_board[pos_and_num / N] = pos_and_num % N + 1;
    }
    print_board(answer_board, N, fp);
    free(answer_board);
}

int dancing_links(int *board, int n, FILE *fp) {
    int N = n * n, i;
    int *col_ids, *row_ids, *convert_table;
    int *dlx;
    int** valid_candidates = (int**) malloc(N * N * sizeof(int*));
    for (i = 0; i < N * N; ++ i)
        valid_candidates[i] = calloc(N, sizeof(int));

    initial_check(board, n, valid_candidates);
    int num_elems = convert_matrix(board, valid_candidates, n, &col_ids, &row_ids, &convert_table);
    int dlx_ncnt = build_dancing_links(col_ids, row_ids, num_elems, &dlx);
    printf("Number of nodes in dancing links: %d\n", dlx_ncnt);

    int *ansnow = calloc(N * N, sizeof(int));
    int num_covered = 0;
    int ret = exact_cover(dlx, dlx + 4 * dlx_ncnt, dlx_ncnt, ansnow, &num_covered, 0);
    
    convert_answer_print(ansnow, convert_table, N, fp);
    
    free(dlx);
    free(ansnow); free(convert_table);
    free(col_ids); free(row_ids);
    for (i = 0; i < N * N; ++ i) free(valid_candidates[i]);
    free(valid_candidates);
    return ret;
}