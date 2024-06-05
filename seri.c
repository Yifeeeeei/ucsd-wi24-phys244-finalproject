#include "models.h"
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#define SEQ_LEN 6
#define TKN_DIM 3
#define KEY_DIM 2
#define VAL_DIM 4
#define NUM_HEAD 16

struct AttentionHead createAttentionHead(int rank)
{
    // currently, rank is not used. it's just a dummy parameter
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
    // MPI initialization
    int numTasks;
    int rank;
    // printf("A Starting...\n");
    // printf("B numTasks = %d\n", numTasks);;
    // printf("D rank = %d\n", rank);
    // printf("E MASTER = %d\n", MASTER);

    // getting data
    printf("master starting\n");
    struct Matrix sequence = getSequence();
    // send the data to each worker
    struct Matrix finalResult = createZeroMatrix(VAL_DIM, SEQ_LEN);

    double start_time = time(NULL);

    for (int i = 1; i < NUM_HEAD; i++)
    {
        struct AttentionHead attentionHead = createAttentionHead(i);
        // sendAttentionHead(i, &attentionHead);
        // sendSequence(i, &sequence);
        struct Matrix result = attention(&attentionHead, &sequence);
        matrixAdd(&finalResult, &result);
    }
    double end_time = time(NULL);

    printf("Serial: Attention processing time = %f seconds.\n", difftime(end_time, start_time));
    // receive the results from each worker

    return 0;
}