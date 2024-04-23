#include "models.h"
#include "mpi.h"
#include <stdlib.h>
#include <stdio.h>

#define MASTER 0
#define SEQ_LEN 6
#define TKN_DIM 3
#define KEY_DIM 2
#define VAL_DIM 4

struct AttentionHead createAttentionHead(int rank)
{
    // currently, rank is not used. it's just a dummy parameter
    struct AttentionHead attentionHead;
    attentionHead.wQuery = createRandomMatrix(KEY_DIM, TKN_DIM);
    attentionHead.wKey = createRandomMatrix(KEY_DIM, TKN_DIM);
    attentionHead.wValue = createRandomMatrix(VAL_DIM, TKN_DIM);
    return attentionHead;
}
void sendAttentionHead(int dest, struct AttentionHead *attentionHead)
{
    MPI_Send(attentionHead->wQuery.data, KEY_DIM * TKN_DIM, MPI_FLOAT, dest, 0, MPI_COMM_WORLD);
    MPI_Send(attentionHead->wKey.data, KEY_DIM * TKN_DIM, MPI_FLOAT, dest, 0, MPI_COMM_WORLD);
    MPI_Send(attentionHead->wValue.data, VAL_DIM * TKN_DIM, MPI_FLOAT, dest, 0, MPI_COMM_WORLD);
}

struct AttentionHead receiveAttentionHead(int source)
{
    struct AttentionHead attentionHead;
    attentionHead.wQuery = createMatrixFrom1DArray(KEY_DIM, TKN_DIM, NULL);
    attentionHead.wKey = createMatrixFrom1DArray(KEY_DIM, TKN_DIM, NULL);
    attentionHead.wValue = createMatrixFrom1DArray(VAL_DIM, TKN_DIM, NULL);
    MPI_Recv(attentionHead.wQuery.data, KEY_DIM * TKN_DIM, MPI_FLOAT, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(attentionHead.wKey.data, KEY_DIM * TKN_DIM, MPI_FLOAT, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(attentionHead.wValue.data, VAL_DIM * TKN_DIM, MPI_FLOAT, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    return attentionHead;

    // return createAttentionHead(source);
}

void sendSequence(int dest, struct Matrix *sequence)
{
    MPI_Send(sequence->data, TKN_DIM * SEQ_LEN, MPI_FLOAT, dest, 0, MPI_COMM_WORLD);
}

struct Matrix receiveAttentionResult(int source)
{
    // the master is expection a float matrix of size VAL_DIM * SEQ_LEN from the worker
    struct Matrix result = createMatrixFrom1DArray(VAL_DIM, SEQ_LEN, NULL);
    MPI_Recv(result.data, VAL_DIM * SEQ_LEN, MPI_FLOAT, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    return result;

    // return createRandomMatrix(VAL_DIM, SEQ_LEN);
}

void sendAttentionResult(int dest, struct Matrix *result)
{
    MPI_Send(result->data, VAL_DIM * SEQ_LEN, MPI_FLOAT, dest, 0, MPI_COMM_WORLD);
}

struct Matrix getSequence()
{
    struct Matrix sequence = createRandomMatrix(TKN_DIM, SEQ_LEN);
    return sequence;
}

struct Matrix receiveSequence(int source)
{
    struct Matrix sequence = createMatrixFrom1DArray(TKN_DIM, SEQ_LEN, NULL);
    MPI_Recv(sequence.data, TKN_DIM * SEQ_LEN, MPI_FLOAT, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    return sequence;

    // return createRandomMatrix(TKN_DIM, SEQ_LEN);
}

int main(int argc, char *argv[])
{
    // MPI initialization
    int numTasks = 4;
    int rank = 1;
    int numWorkers = numTasks - 1;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numTasks);
    if (numTasks < 2)
    {
        printf("Need at least 2 processes\n");
        // MPI_Finalize();
        return 0;
    }
    printf("Rank %d of %d starting...\n", rank, numTasks);
    numWorkers = numTasks - 1;

    if (rank == MASTER)
    {
        // getting data
        printf("master starting\n");
        struct Matrix sequence = getSequence();
        // send the data to each worker
        for (int i = 1; i <= numWorkers; i++)
        {
            struct AttentionHead attentionHead = createAttentionHead(i);
            sendAttentionHead(i, &attentionHead);
            sendSequence(i, &sequence);
        }
        // receive the results from each worker
        for (int i = 1; i <= numWorkers; i++)
        {
            struct Matrix result = receiveAttentionResult(i);
            printMatrix(&result);
        }
        printf("master ending\n");
    }
    else
    {
        // Worker
        struct AttentionHead attentionHead = receiveAttentionHead(MASTER);
        struct Matrix sequence = receiveSequence(MASTER);
        struct Matrix result = attention(&attentionHead, &sequence);
        sendAttentionResult(MASTER, &result);
    }
    // MPI_Finalize();
    return 0;
}