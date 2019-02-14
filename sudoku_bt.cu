/*
 *  Parallel Sudoku solver using backtrack algorithm
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
int cpu_initial_search(int n, int *boards, int space_size);
__global__ void backtrack_kernel(int n, int *ans, int *ans_found);

int* read_board(const char* file_name, int* N) {
    FILE* fp = fopen(file_name, "r");
    int* board = NULL;

    fscanf(fp, "%d", N);
    int total = *N * *N, i;
    board = (int*) calloc(total, sizeof(int));
    for (i = 0; i < total; ++ i)
        fscanf(fp, "%d", board + i);
    return board;
}


int main(int argc, char* argv[]) {

    if (argc < 5) {
        printf("Usage:\n");
        printf("./sudoku_bt <input> <output> <expand> <block_size>\n");
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

    int *boards = (int*) calloc(N * N * expand, sizeof(int));
    memcpy(boards, board, N * N * sizeof(int));
    //print_board(boards, N, stdout);
    int answer_found = cpu_initial_search(n, boards, expand);

    /*int i;
    for (i = 0; i < expand; ++ i)
        print_board(boards + i * N * N, N, stdout);*/
    
    if (answer_found == 1) {
        print_board(boards, N, fp);
    } else if (answer_found == 0) {
        printf("Initial search finished.\n");

        // malloc arrays for cuda
        int *boards_d = 0, *ans_found_d = 0;
        #define errck error_check(__LINE__, 2, boards_d, ans_found_d)
        cudaMalloc((void**) &boards_d, N * N * expand * sizeof(int)); errck;
        cudaMemcpy(boards_d, boards, N * N * expand * sizeof(int), cudaMemcpyHostToDevice); errck;
        cudaMalloc((void**) &ans_found_d, sizeof(int)); errck;
        cudaMemset(ans_found_d, -1, sizeof(int)); errck;

        dim3 grid_dim(expand / tile_size);
        //dim3 block_dim(N, N);
        dim3 block_dim(tile_size);
        backtrack_kernel<<<grid_dim, block_dim, 2 * tile_size * N * N * sizeof(int)>>>(n, boards_d, ans_found_d); errck;

        cudaMemcpy(&answer_found, ans_found_d, sizeof(int), cudaMemcpyDeviceToHost); errck;
        printf("GPU search finished. %d\n", answer_found);
        
        if (answer_found >= 0) {
            cudaMemcpy(boards, boards_d + answer_found * N * N, N * N * sizeof(int), cudaMemcpyDeviceToHost); errck;
            answer_found = 1;
            print_board(boards, N, fp);
        }

        cudaFree(boards_d);
        cudaFree(ans_found_d);
    }

    free(boards);
    return answer_found;

}

// initial search on CPU with (modified) breadth first search
// expand_cnt is the size of expected nodes
// array boards should be multiple of (N * N * expand_cnt)
int cpu_initial_search(int n, int *boards, int expand_cnt) {
    int N = n * n;
    int chunk_size = N * N;
    int head = -1, tail = 1;
    int empty_slot = expand_cnt - 1;
    int i;

    // initialize every slot as empty
    for (i = 1; i < expand_cnt; ++ i)
        boards[i * N * N] = -1;

    // expand search tree
    while (empty_slot > 0 && empty_slot < expand_cnt) {
        // advance head pointer
        head ++;
        if (head == expand_cnt) head = 0;
        int *board_now = boards + head * chunk_size;
        if (board_now[0] == -1) continue;

        // find the first empty location
        int nowp;
        for (nowp = 0; nowp < N * N && board_now[nowp] != 0; ++ nowp);
        if (nowp == N * N) {
            // answer found
            memcpy(boards, board_now, chunk_size * sizeof(int));
            return 1;
        }

        // reserve a value for in-place modification
        // which modifies the board at its original place to save one copy operation
        // this can also be used as flag of answer found or dead node
        int reserve = 0;
        for (i = 1; i <= N; ++ i)
            if (check_partial_board(board_now, n, nowp, i)) {
                // fill in the reserved value
                if (reserve == 0)
                    reserve = i * N * N + nowp;
                else {
                    // find an empty slot
                    if (empty_slot == 0) return 0;
                    int *board_new = boards + tail * chunk_size;
                    while (board_new[0] != -1) {
                        tail ++;
                        if (tail == expand_cnt) tail = 0;
                        board_new = boards + tail * chunk_size;
                    }
                    empty_slot --;
                    // copy and modify the board
                    memcpy(board_new, board_now, chunk_size * sizeof(int));
                    board_new[nowp] = i;
                }
            }
        if (reserve == 0) {
            // dead node
            board_now[0] = -1; // mark board_now == -1 to indicate this slot is empty
            empty_slot ++;
        } else {
            // in-place modification
            board_now[reserve % (N * N)] = reserve / (N * N);
        }
            
    }
    return 0;
}

__device__ int check_partial_board_d(int* board, int n, int p, int num) {
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

__global__ void backtrack_kernel(int n, int *ans_all, int *ans_found) {

    int N = n * n, i;
    int task_id = blockIdx.x * blockDim.x + threadIdx.x;
    // use shared memory
    extern __shared__ int shared_board[];
    int offset = threadIdx.x * N * N;
    for (i = 0; i < N * N; ++ i)
        shared_board[i + offset] = ans_all[task_id * N * N + i];
    int *board = shared_board + offset;
    int *stack = shared_board + blockDim.x * N * N + offset;

    int last_op = 0;    // 0 - push stack, 1 - pop stack
    int top = 0, nowp = 0;
    while (*ans_found == -1) {
        int num_to_try;
        if (last_op == 0) {
            // push stack
            for (;nowp < N * N && board[nowp] != 0; ++ nowp);
            // first check if the board is filled
            if (nowp == N * N) {
                // answer found
                int old = atomicCAS(ans_found, -1, task_id);
                if (old == -1) {
                    // copy back to global memory
                    for (i = 0; i < N * N; ++ i)
                        ans_all[task_id * N * N + i] = board[i];
                }
                break;
            }
            // else initialize the number to try
            num_to_try = 1;
        } else {
            // read stack top and restore
            int stack_num = stack[top];
            nowp = stack_num % (N * N);
            num_to_try = board[nowp] + 1;
        }

        // find next valid number
        for (;num_to_try <= N; ++ num_to_try)
            if (check_partial_board_d(board, n, nowp, num_to_try)) {
                // push stack
                stack[top ++] = nowp;
                // move to next location
                board[nowp] = num_to_try;
                last_op = 0;
                break;
            }
        if (num_to_try > N) {
            // pop stack
            if (top == 0) break;
            board[nowp] = 0;
            top --;
            last_op = 1;
        }
    }
}