#include "models.h"
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#define SEQ_LEN 60
#define TKN_DIM 30
#define KEY_DIM 20
#define VAL_DIM 40
#define NUM_HEAD 8

struct AttentionHead createAttentionHead(int rank)
{
    struct AttentionHead attentionHead;
    attentionHead.wQuery = createRandomMatrix(KEY_DIM, TKN_DIM);
    attentionHead.wKey = createRandomMatrix(KEY_DIM, TKN_DIM);
    attentionHead.wValue = createRandomMatrix(VAL_DIM, TKN_DIM);
    attentionHead.mask = createRandomMatrix(SEQ_LEN, SEQ_LEN);
    return attentionHead;
}

struct Matrix getSequence()
{
    struct Matrix sequence = createRandomMatrix(TKN_DIM, SEQ_LEN);
    return sequence;
}

int main(int argc, char *argv[])
{
    printf("master starting\n");
    struct Matrix sequence = getSequence();
    struct Matrix finalResult = createZeroMatrix(VAL_DIM, SEQ_LEN);

    clock_t start_time = clock();

    for (int i = 1; i < NUM_HEAD; i++)
    {
        struct AttentionHead attentionHead = createAttentionHead(i);
        struct Matrix result = attention(&attentionHead, &sequence);
        matrixAdd(&finalResult, &result);
    }

    clock_t end_time = clock();
    double elapsed_time = ((double) (end_time - start_time)) / CLOCKS_PER_SEC;

    printf("Serial: Attention processing time = %f seconds.\n", elapsed_time);

    return 0;
}