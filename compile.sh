gcc main.c methods.c utils.c -o main -lm -g
nvcc sudoku_dlx.cu methods.c utils.c -o sudoku_dlx -lm -arch=sm_35 -g
nvcc sudoku_bt.cu methods.c utils.c -o sudoku_bt -lm -arch=sm_35 -g -G
nvcc sudoku_obt.cu methods.c utils.c -o sudoku_obt -lm -arch=sm_35 -g -G