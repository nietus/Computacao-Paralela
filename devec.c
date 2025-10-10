// 1. not vectorized: control flow in loop
//    -> O laço contém estruturas de controle de fluxo (if/else) que impedem a vetorização
// 2. not vectorized: not enough data-refs in basic block
//    -> O bloco básico não tem referências de dados suficientes para vetorização
// 3. not vectorized: no vectype for stmt
//    -> O compilador não consegue determinar o tipo de vetor para a operação
// 4. not vectorized: no grouped stores in basic block
//    -> As operações de armazenamento não podem ser agrupadas para vetorização
// 5. not vectorized: not suitable for gather load
//    -> O acesso à memória é não sequencial, exigindo operações de gather não suportadas
// 6. not vectorized: number of iterations cannot be computed
//    -> O compilador não consegue determinar o número de iterações em tempo de compilação
// 7. not vectorized: bad loop form
//    -> A forma do laço não é adequada para vetorização

// Extraído do seguinte comando:
// PS C:\Users\anton\Downloads\computacaoparalela> gcc -O3 -fopt-info-vec-missed devec.c -o devec
// devec.c:14:5: note: not vectorized: control flow in loop.
// devec.c:14:5: note: bad loop form.
// devec.c:14:5: note: not vectorized: not enough data-refs in basic block.
// devec.c:49:1: note: not vectorized: not enough data-refs in basic block.
// devec.c:49:1: note: not vectorized: no vectype for stmt: pretmp_79 = sum;
//  scalar_type: int
// devec.c:49:1: note: not vectorized: not enough data-refs in basic block.
// devec.c:25:12: note: not vectorized: no vectype for stmt: _14 = *_13;
//  scalar_type: int
// devec.c:25:12: note: not vectorized: no vectype for stmt: _17 = *_16;
//  scalar_type: int
// devec.c:25:12: note: not vectorized: no vectype for stmt: *_10 = _18;
//  scalar_type: int
// devec.c:25:12: note: not vectorized: no grouped stores in basic block.
// devec.c:26:18: note: not vectorized: no vectype for stmt: *_24 = _25;
//  scalar_type: int
// devec.c:26:18: note: not vectorized: not enough data-refs in basic block.
// devec.c:27:19: note: not vectorized: not enough data-refs in basic block.
// devec.c:28:18: note: not vectorized: no vectype for stmt: _28 = *_16;
//  scalar_type: int
// devec.c:28:18: note: not vectorized: no vectype for stmt: *_27 = _29;
//  scalar_type: int
// devec.c:28:18: note: not vectorized: no grouped stores in basic block.
// devec.c:30:18: note: not vectorized: no vectype for stmt: _32 = *_16;
//  scalar_type: int
// devec.c:30:18: note: not vectorized: no vectype for stmt: *_31 = _33;
//  scalar_type: int
// devec.c:30:18: note: not vectorized: no grouped stores in basic block.
// devec.c:45:12: note: not vectorized: no vectype for stmt: *ptr_42 = _43;
//  scalar_type: int
// devec.c:45:12: note: not vectorized: not enough data-refs in basic block.
// devec.c:46:18: note: not vectorized: no vectype for stmt: _46 = *_16;
//  scalar_type: int
// devec.c:46:18: note: not vectorized: no vectype for stmt: _48 = *_47;
//  scalar_type: int
// devec.c:46:18: note: not vectorized: no vectype for stmt: *_10 = _49;
//  scalar_type: int
// devec.c:46:18: note: not vectorized: no grouped stores in basic block.
// devec.c:14:5: note: not vectorized: not enough data-refs in basic block.
// devec.c:14:5: note: not vectorized: no vectype for stmt: sum = sum_lsm.13_71;
//  scalar_type: int
// devec.c:14:5: note: not vectorized: not enough data-refs in basic block.
// devec.c:69:5: note: not vectorized: number of iterations cannot be computed.
// devec.c:69:5: note: bad loop form.
// devec.c:58:10: note: not vectorized: no vectype for stmt: n.5_8 = n;
//  scalar_type: int
// devec.c:58:10: note: not vectorized: not enough data-refs in basic block.
// devec.c:59:10: note: not vectorized: no vectype for stmt: n.5_13 = n;
//  scalar_type: int
// devec.c:59:10: note: not vectorized: not enough data-refs in basic block.
// devec.c:60:10: note: not vectorized: no vectype for stmt: n.5_18 = n;
//  scalar_type: int
// devec.c:60:10: note: not vectorized: not enough data-refs in basic block.
// devec.c:62:18: note: not vectorized: not enough data-refs in basic block.
// devec.c:68:5: note: not vectorized: not enough data-refs in basic block.
// devec.c:69:5: note: not vectorized: no vectype for stmt: n.5_62 = n;
//  scalar_type: int
// devec.c:69:5: note: not vectorized: not enough data-refs in basic block.
// devec.c:70:16: note: not vectorized: not enough data-refs in basic block.
// devec.c:71:16: note: not vectorized: no vectype for stmt: *_33 = _36;
//  scalar_type: int
// devec.c:71:16: note: not vectorized: not enough data-refs in basic block.
// devec.c:69:5: note: not vectorized: no vectype for stmt: *_38 = _41;
//  scalar_type: int
// devec.c:69:5: note: not vectorized: no vectype for stmt: *_43 = 0;
//  scalar_type: int
// devec.c:69:5: note: not vectorized: no vectype for stmt: n.5_30 = n;
//  scalar_type: int
// devec.c:69:5: note: not vectorized: no grouped stores in basic block.
// devec.c:79:5: note: not vectorized: no vectype for stmt: n.5_47 = n;
//  scalar_type: int
// devec.c:79:5: note: not vectorized: no vectype for stmt: _52 = *_51;
//  scalar_type: int
// devec.c:79:5: note: not vectorized: no grouped stores in basic block.
// devec.c:51:5: note: not vectorized: not enough data-refs in basic block.

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// 1. Função com loop que contém dependências de dados
void process_data(int n, int *a, int *b, int *c) {
    // 2. Loop com dependência verdadeira
    for (int i = 1; i < n; i++) {
        // 3. Dependência RAW (Read-After-Write) entre iterações
        a[i] = a[i-1] + b[i];
        
        // 4. Acesso não linear à memória (strided access)
        int idx = (i * 7) % n;
        
        // 5. Operação condicional com controle de fluxo complexo
        if (a[i] % 3 == 0) {
            c[i] = a[i] * 2;
        } else if (a[i] % 3 == 1) {
            c[i] = b[i] + 1;
        } else {
            c[i] = a[i] + b[i];
        }
        
        // 6. Operação de redução com variável estática
        static int sum = 0;
        sum += c[i];
        
        // 7. Acesso indireto através de ponteiro
        int *ptr = &b[idx];
        *ptr = c[i] + i;
        
        // 8. Operação que pode causar aliasing
        if (i > n/2) {
            a[i] = b[i] + c[i-1];
        }
    }
}

int main() {
    // Tamanho do array - conhecido apenas em tempo de execução
    int n;
    printf("Digite o tamanho do array: ");
    scanf("%d", &n);
    
    // Alocação dinâmica para evitar otimizações
    int *a = (int*)malloc(n * sizeof(int));
    int *b = (int*)malloc(n * sizeof(int));
    int *c = (int*)malloc(n * sizeof(int));
    
    if (!a || !b || !c) {
        fprintf(stderr, "Falha na alocação de memória\n");
        return 1;
    }
    
    // Inicialização com valores aleatórios
    srand(time(NULL));
    for (int i = 0; i < n; i++) {
        a[i] = rand() % 100;
        b[i] = rand() % 100;
        c[i] = 0;
    }
    
    process_data(n, a, b, c);
    
    printf("Resultado final: %d\n", c[n-1]);
    
    free(a);
    free(b);
    free(c);
    
    return 0;
}
