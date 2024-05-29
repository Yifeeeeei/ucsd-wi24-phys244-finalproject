struct Matrix
{
    int rowNum;
    int colNum;
    float *data;
};
struct AttentionHead
{
    struct Matrix wQuery;
    struct Matrix wKey;
    struct Matrix wValue;
};

struct Matrix matrixMultiply(struct Matrix *matrix1, struct Matrix *matrix2);
struct Matrix createMatrixFrom1DArray(int rowNum, int colNum, float data[]);
struct Matrix createRandomMatrix(int rowNum, int colNum);
struct Matrix createZeroMatrix(int rowNum, int colNum);
void printMatrix(struct Matrix *matrix);
struct Matrix getQuery(struct AttentionHead *attentionHead, struct Matrix *input);
struct Matrix getKey(struct AttentionHead *attentionHead, struct Matrix *input);
struct Matrix getValue(struct AttentionHead *attentionHead, struct Matrix *input);
struct Matrix rowWiseSoftmax(struct Matrix *input);
struct Matrix transpose(struct Matrix input);
struct Matrix attention(struct AttentionHead *attentionHead, struct Matrix *inputSequence);