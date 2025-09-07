gcc tarefa9.c -o tarefa9 -fopenmp
time ./tarefa9

gcc -fopenmp -O3 -o sieve tarefa10.c -lm
perf stat -e cpu-cycles,instructions,cache-references,cache-misses,branches,branch-misses,task-clock ./sieve 1
