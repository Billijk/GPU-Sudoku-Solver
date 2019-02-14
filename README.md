# GPU-Sudoku-Solver
Project of NYU CSCI-GA 3033-004 Graphics Processing Units (GPUs): Architecture and Programming

In this project, I developed a Sudoku Solver running on GPU, which support 4x4, 9x9, and 16x16 Sudoku puzzles. The code is written in C and CUDA C.
Five different algorithms are implemented, which are Brute Force backtracking (sequential version), Dancing Links X (sequencial version), simple parallel Brute Force backtracking, optimized parallel Brute Force backtracking, and parallel Dancing Links X.

Algorithm details and experiment analysis are provided in the [report](report_sudoku.pdf).

## How to run the code
This code requires CUDA-9.1 to compile, and should be run on an Nvidia GPU with compute capability 3.5 or higher.

To compile the code, run `sh compile.sh`. This will generate four executables, which are solvers using different algorithms.

#### Brute Force Backtracking (sequential version)
`./main <input> <output> 0`

#### Dancing Links X (sequencial version)
`./main <input> <output> 1`

#### Simple Parallel Brute Force Backtracking
`./sudoku_bt <input> <output> <expand> <block_size>`
where 'expand' is the number of search trees expanded in initial CPU search, and 'block_size' is the size of a 1-D CUDA block.

#### Optimized Parallel Brute Force Backtracking
`./sudoku_obt <input> <output> <expand>`

#### Parallel Dancing Links X
`./sudoku_dlx <input> <output> <expand> <block_size>`

Some sample input puzzles are provided in directory inputs/. You can also provide custom puzzles in the following format:
```
4
0 0 0 0 
0 0 0 0 
0 1 4 0 
3 0 0 1 
```
The first line is the size of the puzzle, next is the puzzle, with 0 indicating the unseen tiles.

## Experiment results
The first two sample puzzles are 4x4 in size and too simple, so results on them are not compared.

| Puzzle ID | BF             | DLX       | SPBF    | OPBF      | PDLX   |
|:---------:|----------------|-----------|---------|-----------|--------|
|     3     | 0.011          | **0.010** | 0.284   | 0.292     | 0.288  |
|     4     | 0.357          | **0.057** | 0.400   | 0.358     | 0.531  |
|     5     | **0.383**      | 0.493     | 0.596   | 0.505     | 0.930  |
|     6     | 1.114          | 0.815     | 1.813   | **0.668** | 1.532  |
|     7     | **0.160**      | 0.589     | 3.295   | 1.084     | 1.685  |
|     8     | 0.495          | 1.539     | 1.049   | **0.422** | 5.213  |
|     9     | ~300           | ~75       | 1.683   | **1.445** | 1.892  |