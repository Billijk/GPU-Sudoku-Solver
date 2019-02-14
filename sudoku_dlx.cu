/*
 *  Parallel Sudoku solver using dancing links algorithm
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <math.h>

// declare C include here
extern "C" { 
    #include "methods.h" 
}

int solve(int *board, int n, int expand, int tile_size, FILE *fp);
int cpu_initial_search(int n, int dlx_size, int *ans, int *dlxs, int *dlx_props, int space_size);
__global__ void exact_cover_kernel(int *dlx, int *dlx_props, int dlx_size, int N, int *ans, int *ans_found);

int* read_board(const char* file_name, int* N) {
    FILE* fp = fopen(file_name, "r");
    int* board = NULL;

    fscanf(fp, "%d", N);
    int total = *N * *N;
    board = (int*) calloc(total, sizeof(int));
    for (int i = 0; i < total; ++ i)
        fscanf(fp, "%d", board + i);
    return board;
}


int main(int argc, char* argv[]) {

    if (argc < 5) {
        printf("Usage:\n");
        printf("./sudoku_dlx <input> <output> <expand> <block_size>\n");
        exit(-1);
    }

    int N;
    int* board = read_board(argv[1], &N);
    int n = sqrt((double)N);
    printf("Start to solve a %d x %d Sudoku.\n", N, N);

    FILE* fp = fopen(argv[2], "w");
    int expand = (int) atoi(argv[3]);
    int tile_size = (int) atoi(argv[4]);
    int ans = solve(board, n, expand, tile_size, fp);

    if (ans == 1) printf("An answer is found and saved to %s.\n", argv[2]);
    else printf("No answer is found.\n");

    free(board);
    return 0;
}

void error_check(int line_number, int arg_count, ...) {
    cudaError_t err=cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s in %s at line %d\n", cudaGetErrorString(err), line_number);
        va_list ap;
        va_start(ap, arg_count);
        int i;
        for (i = 0; i < arg_count; ++ i) {
            int *arr = va_arg(ap, int*);
            if (arr != 0) cudaFree(arr);
        }
        va_end(ap);
        exit(-1);
    }
}

// solver function
int solve(int* board, int n, int expand, int tile_size, FILE* fp) {
    int N = n * n;

    int *col_ids, *row_ids, *convert_table;
    int *dlx;
    int** valid_candidates = (int**) malloc(N * N * sizeof(int*));
    for (int i = 0; i < N * N; ++ i)
        valid_candidates[i] = (int*) calloc(N, sizeof(int));

    initial_check(board, n, valid_candidates);
    int num_elems = convert_matrix(board, valid_candidates, n, &col_ids, &row_ids, &convert_table);
    int dlx_ncnt = build_dancing_links(col_ids, row_ids, num_elems, &dlx);
    printf("Number of nodes in dancing links: %d\n", dlx_ncnt);

    int *ans = (int*) calloc(N * N * expand, sizeof(int));
    int *dlxs = (int*) calloc(dlx_ncnt * 4 * expand, sizeof(int));
    int *dlx_props = dlx + 4 * dlx_ncnt;
    for (int i = 0; i < dlx_ncnt * 4; ++ i) dlxs[i] = dlx[i];
    int answer_found = cpu_initial_search(N, dlx_ncnt, ans, dlxs, dlx_props, expand);
    if (answer_found == 1) {
        for (int i = 0; i < N * N; ++ i) ans[i] = dlx_props[ans[i] + dlx_ncnt]; // convert to row numbers
        convert_answer_print(ans, convert_table, N, fp);
    } else if (answer_found == 0) {
        printf("Initial search finished.\n");

        // malloc arrays for cuda
        int *ans_d = 0, *dlxs_d = 0, *dlx_props_d = 0, *ans_found_d = 0;
        #define errck error_check(__LINE__, 4, ans_d, dlxs_d, dlx_props_d, ans_found_d)
        cudaMalloc((void**) &ans_d, N * N * expand * sizeof(int)); errck;
        cudaMemcpy(ans_d, ans, N * N * expand * sizeof(int), cudaMemcpyHostToDevice); errck;
        cudaMalloc((void**) &dlxs_d, dlx_ncnt * 4 * expand * sizeof(int)); errck;
        cudaMemcpy(dlxs_d, dlxs, dlx_ncnt * 4 * expand * sizeof(int), cudaMemcpyHostToDevice); errck;
        cudaMalloc((void**) &dlx_props_d, dlx_ncnt * 2 * sizeof(int)); errck;
        cudaMemcpy(dlx_props_d, dlx_props, dlx_ncnt * 2 * sizeof(int), cudaMemcpyHostToDevice); errck;
        cudaMalloc((void**) &ans_found_d, sizeof(int)); errck;
        cudaMemset(ans_found_d, -1, sizeof(int)); errck;

        exact_cover_kernel<<<expand / tile_size, tile_size>>>(dlxs_d, dlx_props_d, dlx_ncnt, N, ans_d, ans_found_d); errck;

        cudaMemcpy(&answer_found, ans_found_d, sizeof(int), cudaMemcpyDeviceToHost); errck;
        printf("GPU search finished.\n");
        
        if (answer_found >= 0) {
            cudaMemcpy(ans, ans_d + answer_found * N * N, N * N * sizeof(int), cudaMemcpyDeviceToHost); errck;
            for (int i = 0; i < N * N; ++ i) ans[i] = dlx_props[ans[i] + dlx_ncnt]; // convert to row numbers
            convert_answer_print(ans, convert_table, N, fp);
            answer_found = 1;
        }

        cudaFree(ans_d);
        cudaFree(dlxs_d);
        cudaFree(dlx_props_d);
        cudaFree(ans_found_d);
    }
    
    free(dlxs); free(dlx); free(ans); free(convert_table);
    free(col_ids); free(row_ids);
    for (int i = 0; i < N * N; ++ i) free(valid_candidates[i]);
    free(valid_candidates);
    return answer_found;

}

// initial search on CPU with (modified) breadth first search
// expand_cnt is the size of expected nodes
// array ans should be multiple of (N * N * expand_cnt)
// array dlxs should be multiple of (dlx_size * 4 * expand_cnt)
int cpu_initial_search(int N, int dlx_size, int *ans, int *dlxs, int *dlx_props, int expand_cnt) {
    int chunk_size = dlx_size * 4;
    int head = -1, tail = 1;
    int *up, *down, *left, *right, *col, *row;
    int empty_slot = expand_cnt - 1;

    // initialize every slot as empty
    for (int i = 1; i < expand_cnt; ++ i)
        *(ans + i * N * N) = -1;

    // expand search tree
    while (empty_slot > 0 && empty_slot < expand_cnt) {
        // advance head pointer
        head ++;
        if (head == expand_cnt) head = 0;

        int *dlxs_head = dlxs + head * chunk_size;
        int *ans_head = ans + head * N * N;
        if (*ans_head == -1) continue;

        /*printf("%d %d %d\n", head, tail, empty_slot);
        for (int i = 0; i < expand_cnt; ++ i)
            printf("%d ", *(ans + i * N * N));
        printf("\n\n");*/
    
        // fetch current head node
        UNLOAD(dlxs_head, dlx_props, dlx_size);
        int current_col = right[0];
        int row_id = down[current_col];

        // if this node is dead
        if (row_id == current_col) {
            *ans_head = -1; // mark ans_head == -1 to indicate this slot is empty
            empty_slot ++;
            continue;
        }
        
        // the first element of ans is used to store depth
        // answers are saved in reverse order, so that during DFS searching
        // ans can be stored at same level as stack
        int depth = *ans_head;
        int depth_offset = N * N - depth - 1;
        
        // otherwise expand this node. 
        for (;down[row_id] != current_col; row_id = down[row_id]) {
            // find an empty slot
            int *ans_tail = ans + tail * N * N;
            while (*ans_tail != -1) {
                tail ++;
                if (tail == expand_cnt) tail = 0;
                ans_tail = ans + tail * N * N;
            }
            int *dlxs_tail = dlxs + tail * chunk_size;
            empty_slot --;

            // copy structure to the slot
            for (int i = 0; i < chunk_size; ++ i)
                *(dlxs_tail + i) = *(dlxs_head + i);
            for (int i = 0; i < N * N; ++ i)
                *(ans_tail + i) = *(ans_head + i);
            *(ans_tail + depth_offset) = row_id; // write current answer

            // modify the structure
            remove_column(current_col, dlxs_tail, dlx_props, dlx_size);
            for (int elem = right[row_id]; elem != row_id; elem = right[elem])
                remove_column(col[elem], dlxs_tail, dlx_props, dlx_size);

            // if this search already finds an answer
            if (*(dlxs_tail + 3 * dlx_size) == 0) {
                // copy the answer to the begining locations of ans
                for (int i = 0; i < N * N; ++ i)
                    *(ans + i) = *(ans_tail + i);
                return 1;
            }

            // if the first element is not occupied, save length here
            if (depth_offset != 0) *ans_tail = depth + 1;
            if (empty_slot == 0) break;
        }
        if (down[row_id] != current_col && empty_slot == 0) break;
        // reuse this node to avoid a copy
        *(ans_head + depth_offset) = row_id; // write current answer
        remove_column(current_col, dlxs_head, dlx_props, dlx_size);
        for (int elem = right[row_id]; elem != row_id; elem = right[elem])
            remove_column(col[elem], dlxs_head, dlx_props, dlx_size);
        if (depth_offset != 0) *ans_head = depth + 1;
        if (right[0] == 0) {
            // copy the answer to the begining locations of ans
            for (int i = 0; i < N * N; ++ i)
                *(ans + i) = *(ans_head + i);
            return 1;
        }

    }
    if (empty_slot == 0) return 0;
    else return -1;
}

// kernel functions
__device__ void remove_column_d(int id, int *dlx, int *dlx_props, int dlx_size) {
    int *up, *down, *left, *right, *col, *row;
    UNLOAD(dlx, dlx_props, dlx_size);
    
    // first detach the column indicator
    right[left[id]] = right[id];
    left[right[id]] = left[id];

    // find every row of this column
    for (int row_id = down[id]; row_id != id; row_id = down[row_id]) {
        // find every element in that row
        for (int elem = right[row_id]; elem != row_id; elem = right[elem]) {
            // detach that element
            down[up[elem]] = down[elem];
            up[down[elem]] = up[elem];
        }
    }
}
__device__ void restore_column_d(int id, int *dlx, int *dlx_props, int dlx_size) {
    int *up, *down, *left, *right, *col, *row;
    UNLOAD(dlx, dlx_props, dlx_size);

    // first detach the column indicator
    right[left[id]] = id;
    left[right[id]] = id;

    // find every row of this column
    for (int row_id = down[id]; row_id != id; row_id = down[row_id]) {
        // find every element in that row
        for (int elem = right[row_id]; elem != row_id; elem = right[elem]) {
            // attach that element
            down[up[elem]] = elem;
            up[down[elem]] = elem;
        }
    }
}

__global__ void exact_cover_kernel(int *dlxs, int *dlx_props, int dlx_size, int N, int *ans_all, int *ans_found) {

    int *up, *down, *left, *right, *col, *row;
    int task_id = threadIdx.x + blockIdx.x * blockDim.x;
    int *dlx = dlxs + task_id * dlx_size * 4;
    int *ans = ans_all + task_id * N * N;
    UNLOAD(dlx, dlx_props, dlx_size);

    int last_op = 0;    // 0 - push stack, 1 - pop stack
    int top = 0;
    while (*ans_found == -1) {
        int row_id;
        if (last_op == 0) {
            if (right[0] == 0) {
                // every element has been covered, answer found
                atomicCAS(ans_found, -1, task_id);
                break;
            }

            int current_col = right[0];
            row_id = down[current_col];
            if (row_id == current_col) {
                // this column has not been covered
                if (top == 0) break;
                top --;
                last_op = 1;
                continue;
            }

        } else {

            // read stack top and restore
            row_id = ans[top];
            for (int elem = right[row_id]; elem != row_id; elem = right[elem])
                restore_column_d(col[elem], dlx, dlx_props, dlx_size);
            restore_column_d(col[row_id], dlx, dlx_props, dlx_size);
            row_id = down[row_id];

            // this column has finished iteration
            if (row_id == right[0]) {
                // pop stack
                if (top == 0) break;
                top --;
                continue;
            }

        }

        remove_column_d(col[row_id], dlx, dlx_props, dlx_size);
        for (int elem = right[row_id]; elem != row_id; elem = right[elem])
            remove_column_d(col[elem], dlx, dlx_props, dlx_size);
        
    
        // push stack
        ans[top] = row_id;
        top ++;
        last_op = 0;
        
    }

}