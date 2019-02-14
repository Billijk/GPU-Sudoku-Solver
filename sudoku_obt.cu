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

int solve(int *board, int n, int expand, FILE *fp);
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

    if (argc < 4) {
        printf("Usage:\n");
        printf("./sudoku_obt <input> <output> <expand>\n");
        exit(-1);
    }

    int N;
    int* board = read_board(argv[1], &N);
    int n = sqrt((double)N);
    printf("Start to solve a %d x %d Sudoku.\n", N, N);

    FILE* fp = fopen(argv[2], "w");
    int expand = (int) atoi(argv[3]);
    int ans = solve(board, n, expand, fp);

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
int solve(int* board, int n, int expand, FILE* fp) {
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

        dim3 grid_dim(expand);
        dim3 block_dim(N, N);
        //dim3 block_dim(tile_size);
        backtrack_kernel<<<grid_dim, block_dim, (N * N + N + 4) * sizeof(int)>>>(n, boards_d, ans_found_d); errck;

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

__global__ void backtrack_kernel(int n, int *ans_all, int *ans_found) {

    int N = n * n;
    // use shared memory
    extern __shared__ int shared[];
    int board_num = ans_all[blockIdx.x * N * N + (threadIdx.x * N) + threadIdx.y];
    int *failed = shared + 4;
    int *stack = shared + N + 4;
    int top = 0;
    // put all locations of empty tiles in stack
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        int i;
        shared[0] = -1;
        for (i = 0; i < N * N; ++ i) {
            if (ans_all[blockIdx.x * N * N + i] == 0)
                stack[top ++] = i;
        }
        stack[top] = -1;
        top = 0;
    }
    int box_now = threadIdx.x / n * n + threadIdx.y / n;
    __syncthreads();

    int last_op = 0;    // 0 - push stack, 1 - pop stack
    while (*ans_found == -1) {
        if (threadIdx.x == 0 && threadIdx.y == 0) {
            int stack_num = stack[top];
            shared[0] = stack_num % (N * N); // communicate nowp
            if (last_op == 0) {
                // first check if the board is filled
                if (stack_num == -1) {
                    // answer found
                    atomicCAS(ans_found, -1, blockIdx.x);
                    shared[0] = -1;
                }
                // else initialize the number to try
                shared[1] = 1;
            } else {
                shared[1] = stack_num / (N * N) + 1;
            }
        }
        if (threadIdx.y == 0) failed[threadIdx.x] = 0;
        __syncthreads();
        
        // find next valid number
        int nowp = shared[0], i = shared[1];
        if (nowp == -1) break;
        int num_to_try = i;
        if (ROW(nowp, N) == threadIdx.x || COL(nowp, N) == threadIdx.y || BOX(nowp, n) == box_now) {
            for (; i <= N; ++ i) if (i == board_num) failed[i - 1] = 1;
        }
        __syncthreads();

        if (threadIdx.x == 0 && threadIdx.y == 0) {
            for (i = num_to_try; i <= N && failed[i - 1] != 0; ++ i);
            shared[2] = nowp;   // record in shared memory for communication
            if (i <= N) {
                // push stack
                stack[top ++] = i * N * N + nowp;
                shared[3] = i;
                last_op = 0;
            } else {
                // pop stack
                if (top == 0) shared[2] = -1;
                stack[top --] = nowp;
                shared[3] = 0;
                last_op = 1;
            }
        }
        __syncthreads();
        if (shared[2] == -1) break;
        // update local register of last step
        if (threadIdx.x * N + threadIdx.y == shared[2])
            board_num = shared[3];

    }

    if (*ans_found == blockIdx.x)
        ans_all[blockIdx.x * N * N + (threadIdx.x * N) + threadIdx.y] = board_num;
}