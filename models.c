#include <cblas.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "models.h"

struct Matrix matrixMultiply(struct Matrix *matrix1, struct Matrix *matrix2)
{
    // struct Matrix result;
    // result.rowNum = matrix1->rowNum;
    // result.colNum = matrix2->colNum;
    // if (matrix1->colNum != matrix2->rowNum)
    // {
    //     printf("Matrix multiplication error: matrix1.colNum != matrix2.rowNum\n");
    //     return result;
    // }
    // result.data = (float *)malloc(sizeof(float) * result.rowNum * result.colNum);
    // for (int i = 0; i < result.rowNum; i++)
    // {
    //     for (int j = 0; j < result.colNum; j++)
    //     {
    //         result.data[i * result.colNum + j] = 0;
    //         for (int k = 0; k < matrix1->colNum; k++)
    //         {
    //             result.data[i * result.colNum + j] += matrix1->data[i * matrix1->colNum + k] * matrix2->data[k * matrix2->colNum + j];
    //         }
    //     }
    // }

    // return result;

    // using open blas
    struct Matrix result;
    if (matrix1->colNum != matrix2->rowNum)
    {
        printf("Matrix multiplication error: matrix1.colNum != matrix2.rowNum\n");
        result.rowNum = 0;
        result.colNum = 0;
        result.data = NULL;
        return result;
    }

    result.rowNum = matrix1->rowNum;
    result.colNum = matrix2->colNum;
    result.data = (float *)malloc(sizeof(float) * result.rowNum * result.colNum);
    if (result.data == NULL)
    {
        printf("Memory allocation failed\n");
        result.rowNum = 0;
        result.colNum = 0;
        return result;
    }

    // Perform matrix multiplication using OpenBLAS
    // C = alpha * A * B + beta * C
    // We want C = A * B, so alpha = 1, beta = 0
    float alpha = 1.0f;
    float beta = 0.0f;
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                result.rowNum, result.colNum, matrix1->colNum,
                alpha, matrix1->data, matrix1->colNum,
                matrix2->data, matrix2->colNum,
                beta, result.data, result.colNum);

    return result;
}

struct Matrix createMatrixFrom1DArray(int rowNum, int colNum, float data[])
{
    struct Matrix matrix;
    matrix.rowNum = rowNum;
    matrix.colNum = colNum;
    matrix.data = malloc(sizeof(float) * rowNum * colNum);
    for (int i = 0; i < rowNum * colNum; i++)
    {
        matrix.data[i] = data[i];
    }
    return matrix;
}

struct Matrix createRandomMatrix(int rowNum, int colNum)
{
    struct Matrix matrix;
    matrix.rowNum = rowNum;
    matrix.colNum = colNum;
    matrix.data = malloc(sizeof(float) * rowNum * colNum);
    // each element bewteen 0 and 1
    for (int i = 0; i < rowNum * colNum; i++)
    {
        matrix.data[i] = (float)rand() / RAND_MAX;
    }
    return matrix;
}

struct Matrix createZeroMatrix(int rowNum, int colNum)
{
    struct Matrix matrix;
    matrix.rowNum = rowNum;
    matrix.colNum = colNum;
    matrix.data = malloc(sizeof(float) * rowNum * colNum);
    // each element bewteen 0 and 1
    for (int i = 0; i < rowNum * colNum; i++)
    {
        matrix.data[i] = 0;
    }
    return matrix;
}

void printMatrix(struct Matrix *matrix)
{
    for (int i = 0; i < matrix->rowNum; i++)
    {
        for (int j = 0; j < matrix->colNum; j++)
        {
            printf("%f ", matrix->data[i * matrix->colNum + j]);
        }
        printf("\n");
    }
}

struct Matrix getQuery(struct AttentionHead *attentionHead, struct Matrix *input)
{
    return matrixMultiply(&attentionHead->wQuery, input);
}
struct Matrix getKey(struct AttentionHead *attentionHead, struct Matrix *input)
{
    return matrixMultiply(&attentionHead->wKey, input);
}
struct Matrix getValue(struct AttentionHead *attentionHead, struct Matrix *input)
{
    return matrixMultiply(&attentionHead->wValue, input);
}

struct Matrix rowWiseSoftmax(struct Matrix *input)
{
    struct Matrix result;
    result.rowNum = input->rowNum;
    result.colNum = input->colNum;
    result.data = malloc(sizeof(float) * input->rowNum * input->colNum);
    for (int i = 0; i < input->rowNum; i++)
    {
        long sum = 0;
        for (int j = 0; j < input->colNum; j++)
        {
            sum += exp(input->data[i * input->colNum + j]);
        }
        for (int j = 0; j < input->colNum; j++)
        {
            result.data[i * input->colNum + j] = exp(input->data[i * input->colNum + j]) / sum;
        }
    }
    return result;
}

struct Matrix transpose(struct Matrix input)
{
    struct Matrix result;
    result.rowNum = input.colNum;
    result.colNum = input.rowNum;
    result.data = malloc(sizeof(float) * result.rowNum * result.colNum);
    for (int i = 0; i < input.rowNum; i++)
    {
        for (int j = 0; j < input.colNum; j++)
        {
            result.data[j * input.rowNum + i] = input.data[i * input.colNum + j];
        }
    }
    return result;
}

struct Matrix attention(struct AttentionHead *attentionHead, struct Matrix *inputSequence)
{
    struct Matrix Q = getQuery(attentionHead, inputSequence);
    struct Matrix QTranspose = transpose(Q);
    struct Matrix K = getKey(attentionHead, inputSequence);
    struct Matrix V = getValue(attentionHead, inputSequence);
    struct Matrix VTranspose = transpose(V);
    struct Matrix scores = matrixMultiply(&QTranspose, &K);
    struct Matrix attentionMatrix = rowWiseSoftmax(&scores);
    if (attentionHead->mask.rowNum > 0 && attentionHead->mask.colNum > 0)
    {
        // apply mask
        for (int i = 0; i < attentionMatrix.rowNum; i++)
        {
            for (int j = 0; j < attentionMatrix.colNum; j++)
            {
                attentionMatrix.data[i * attentionMatrix.colNum + j] *= attentionHead->mask.data[i * attentionMatrix.colNum + j];
            }
        }
    }

    struct Matrix result = matrixMultiply(&attentionMatrix, &VTranspose);
    return transpose(result);
}
// int main()
// {

//     printf("running attentionModels\n");

//     int inputSequenceLength = 6;
//     struct Matrix inputSequence = createRandomMatrix(3, inputSequenceLength);
//     struct AttentionHead attentionHead;
//     attentionHead.wQuery = createRandomMatrix(3, 3);
//     attentionHead.wKey = createRandomMatrix(3, 3);
//     attentionHead.wValue = createRandomMatrix(3, 3);
//     struct Matrix result;
//     result = attention(&attentionHead, &inputSequence);
//     printMatrix(&result);
//     return 0;
// }