// for both input transform and output transform shaders
#define BLOCK_SIZE 64

RWStructuredBuffer<float16_t4>  input             : register(u0);
RWStructuredBuffer<float16_t>   transformedInput  : register(u1);

cbuffer consts : register(b0) {
    uint N, C;
    uint relu;
    uint useBias;
};


void matrixMul_gpu_serial_6x6x6(out float16_t c[6][6], in float16_t a[6][6], in float16_t b[6][6])
{
    [unroll]
    for (int i = 0; i < 6; ++i)
        [unroll]
        for (int j = 0; j < 6; ++j)
        {
            float16_t S = 0;
            [unroll]
            for (int k = 0; k < 6; ++k)
                S += a[i][k] * b[k][j];
            c[i][j] = S;
        }
}

void matrixMul_gpu_serial_6x6x6(out float c[6][6], in float a[6][6], in float b[6][6])
{
    [unroll]
    for (int i = 0; i < 6; ++i)
        [unroll]
        for (int j = 0; j < 6; ++j)
        {
            float S = 0;
            [unroll]
            for (int k = 0; k < 6; ++k)
                S += a[i][k] * b[k][j];
            c[i][j] = S;
        }
}


void matrixMul_gpu_serial_4x6x6(out float16_t c[4][6], in float16_t a[4][6], in float16_t b[6][6])
{
    [unroll]
    for (int i = 0; i < 4; ++i)
        [unroll]
        for (int j = 0; j < 6; ++j)
        {
            float16_t S = 0;
            [unroll]
            for (int k = 0; k < 6; ++k)
                S += a[i][k] * b[k][j];
            c[i][j] = S;
        }
}

void matrixMul_gpu_serial_4x4x6(out float16_t c[4][4], in float16_t a[4][6], in float16_t b[6][4])
{
    [unroll]
    for (int i = 0; i < 4; ++i)
        [unroll]
        for (int j = 0; j < 4; ++j)
        {
            float16_t S = 0;
            [unroll]
            for (int k = 0; k < 6; ++k)
                S += a[i][k] * b[k][j];
            c[i][j] = S;
        }
}

void inputTransform4x4_gpu(out float16_t op[6][6], in const float16_t ip[6][6])
{
    // transform applied to input tile (of size 4x4 - padded up to 6x6)
    const float16_t Bt[6][6] = 
    {
        4,  0, -5,  0, 1, 0,
        0, -4, -4,  1, 1, 0,
        0,  4, -4, -1, 1, 0,
        0, -2, -1,  2, 1, 0,
        0,  2, -1, -2, 1, 0,
        0,  4,  0, -5, 0, 1
    };

    const float16_t B[6][6] =
    {
        4,  0,  0,  0,  0,  0,
        0, -4,  4, -2,  2,  4,
       -5, -4, -4, -1, -1,  0,
        0,  1, -1,  2, -2, -5,
        1,  1,  1,  1,  1,  0,
        0,  0,  0,  0,  0,  1
    };

    float16_t tempIp1[6][6];
    matrixMul_gpu_serial_6x6x6(tempIp1, Bt, ip);
    matrixMul_gpu_serial_6x6x6(op, tempIp1, B);
}

void outputTransform4x4_gpu(out float16_t output[4][4], in const float16_t transformedOutput[6][6])
{
    // transform applied to result
    const float16_t At[4][6] = {
        1, 1, 1, 1, 1, 0,
        0, 1,-1, 2,-2, 0,
        0, 1, 1, 4, 4, 0,
        0, 1,-1, 8,-8, 1
    };

    const float16_t A[6][4] = {
        1, 0, 0, 0,
        1, 1, 1, 1,
        1,-1, 1,-1,
        1, 2, 4, 8,
        1,-2, 4,-8,
        0, 0, 0, 1
    };

    float16_t tempOp[4][6];
    matrixMul_gpu_serial_4x6x6(tempOp, At, transformedOutput);
    matrixMul_gpu_serial_4x4x6(output, tempOp, A);
}


void inputTransform4x4_gpu(out float op[6][6], in const float ip[6][6])
{
    // transform applied to input tile (of size 4x4 - padded up to 6x6)
    const float Bt[6][6] =
    {
        4,  0, -5,  0, 1, 0,
        0, -4, -4,  1, 1, 0,
        0,  4, -4, -1, 1, 0,
        0, -2, -1,  2, 1, 0,
        0,  2, -1, -2, 1, 0,
        0,  4,  0, -5, 0, 1
    };

    const float B[6][6] =
    {
        4,  0,  0,  0,  0,  0,
        0, -4,  4, -2,  2,  4,
       -5, -4, -4, -1, -1,  0,
        0,  1, -1,  2, -2, -5,
        1,  1,  1,  1,  1,  0,
        0,  0,  0,  0,  0,  1
    };

    float tempIp1[6][6];
    matrixMul_gpu_serial_6x6x6(tempIp1, Bt, ip);
    matrixMul_gpu_serial_6x6x6(op, tempIp1, B);
}


// index in input/output tensors
#define INDEX_NCHW(n,c,h,w) ((n)*C*H*W + (c)*H*W + (h)*W + w)

// index in intermediate/temp tensor
// W, H == 6 here! (6x6 transformed blocks)
// N also includes part of dimension (2x2)
#define GemmN (N * 4)
#define TEMP_INDEX_HWNC(h,w,n,c) ((h)*6*GemmN*C + (w)*GemmN*C + (n)*C + c)


[numthreads(BLOCK_SIZE, 1, 1)] 
void InputTransform
(    
    uint3 tid : SV_DispatchThreadID
)
{
    const int H = 8, W = 8;
    int c = tid.x % C;
    int n = tid.x / C;
    if (n > N) return;

    float16_t board[8][8];
    
    // read the board (a row at a time)
    [unroll]
    for (int y = 0; y < 8; y++)
    {
        int index = INDEX_NCHW(n, c, y, 0) / 4;
        float16_t4 r1 = input[index];
        float16_t4 r2 = input[index + 1];
        board[y][0] = r1.x;
        board[y][1] = r1.y;
        board[y][2] = r1.z;
        board[y][3] = r1.w;
        board[y][4] = r2.x;
        board[y][5] = r2.y;
        board[y][6] = r2.z;
        board[y][7] = r2.w;
    }

    // top-left
    {
        float16_t inEl[6][6] = {0,0,0,0,0,0, 0,0,0,0,0,0, 0,0,0,0,0,0, 0,0,0,0,0,0, 0,0,0,0,0,0, 0,0,0,0,0,0};

        [unroll]
        for (int i = 0; i < 5; i++)
            [unroll]
            for (int j = 0; j < 5; j++)
                inEl[i + 1][j + 1] = board[i][j];

        // ii) transform it
        inputTransform4x4_gpu(inEl, inEl);

        // iii) write to output
        [unroll]
        for (int y = 0; y < 6; y++)
            [unroll]
            for (int x = 0; x < 6; x++)
                transformedInput[TEMP_INDEX_HWNC(y, x, n * 4 + 0, c)] = inEl[y][x];
    }

    // top-right
    {
        half inEl[6][6] = { 0,0,0,0,0,0, 0,0,0,0,0,0, 0,0,0,0,0,0, 0,0,0,0,0,0, 0,0,0,0,0,0, 0,0,0,0,0,0 };

        [unroll]
        for (int i = 0; i < 5; i++)
            [unroll]
            for (int j = 0; j < 5; j++)
                inEl[i + 1][j] = board[i][j+3];


        // ii) transform it
        inputTransform4x4_gpu(inEl, inEl);

        // iii) write to output
        [unroll]
        for (int y = 0; y < 6; y++)
            [unroll]
            for (int x = 0; x < 6; x++)
                transformedInput[TEMP_INDEX_HWNC(y, x, n * 4 + 1, c)] = inEl[y][x];
    }


    // bottom-left
    {
        half inEl[6][6] = { 0,0,0,0,0,0, 0,0,0,0,0,0, 0,0,0,0,0,0, 0,0,0,0,0,0, 0,0,0,0,0,0, 0,0,0,0,0,0 };

        [unroll]
        for (int i = 0; i < 5; i++)
            [unroll]
            for (int j = 0; j < 5; j++)
                inEl[i][j + 1] = board[i+3][j];

        // ii) transform it
        inputTransform4x4_gpu(inEl, inEl);
        
        // iii) write to output
        [unroll]
        for (int y = 0; y < 6; y++)
            [unroll]
            for (int x = 0; x < 6; x++)
                transformedInput[TEMP_INDEX_HWNC(y, x, n * 4 + 2, c)] = inEl[y][x];
    }

    // bottom-right
    {
        half inEl[6][6] = { 0,0,0,0,0,0, 0,0,0,0,0,0, 0,0,0,0,0,0, 0,0,0,0,0,0, 0,0,0,0,0,0, 0,0,0,0,0,0 };

        [unroll]
        for (int i = 0; i < 5; i++)
            [unroll]
            for (int j = 0; j < 5; j++)
                inEl[i][j] = board[i+3][j+3];

        // ii) transform it
        inputTransform4x4_gpu(inEl, inEl);

        // iii) write to output
        [unroll]
        for (int y = 0; y < 6; y++)
            [unroll]
            for (int x = 0; x < 6; x++)
                transformedInput[TEMP_INDEX_HWNC(y, x, n * 4 + 3, c)] = inEl[y][x];
    }
}


RWStructuredBuffer<float16_t>  transformedOutput    : register(u0);
RWStructuredBuffer<float16_t4> output               : register(u1);
RWStructuredBuffer<float16_t>  bias                 : register(u2);

[numthreads(BLOCK_SIZE, 1, 1)]
void OutputTransform
(
    uint3 tid : SV_DispatchThreadID
)
{
    const int H = 8, W = 8;

    int k = tid.x % C;      // C is set to K in the constant buffer
    int n = tid.x / C;
    if (n > N) return;

    float16_t board[8][8];
    float16_t b = useBias ? bias[k] : 0;

    [unroll]
    for (int hStart = 0; hStart < 8; hStart += 4)
        [unroll]
        for (int wStart = 0; wStart < 8; wStart += 4)
        {
            //  i) read to per thread registers (for doing output transform)
            int shln = n * 4 + (hStart / 4) * 2 + (wStart / 4);
            float16_t outElTransformed[6][6];
            [unroll]
            for (int y = 0; y < 6; y++)
                [unroll]
                for (int x = 0; x < 6; x++)
                    outElTransformed[y][x] = transformedOutput[TEMP_INDEX_HWNC(y, x, shln, k)];

            // ii) transform it
            float16_t outEl[4][4];
            outputTransform4x4_gpu(outEl, outElTransformed);

            {
                [unroll]
                for (int y = 0; y < 4; y++)
                    [unroll]
                    for (int x = 0; x < 4; x++)
                        board[hStart + y][wStart + x] = outEl[y][x];
            }
        }

    // iii) apply relu and bias
    [unroll]
    for (int y = 0; y < 8; y++)
        [unroll]
        for (int x = 0; x < 8; x++)
        {
            board[y][x] += b;
            if (relu && board[y][x] < 0)
                board[y][x] = 0;
        }


    // iv) write to output
    {
        [unroll]
        for (int y = 0; y < 8; y++)
        {
            int index = INDEX_NCHW(n, k, y, 0) / 4;
            // can possibly use uint4 to write entire row at a time?
            // couldn't find half2 to uint re-interpret functions :(
            // same issue for reads.
            float16_t4 r1;
            float16_t4 r2;
            r1.x = board[y][0];
            r1.y = board[y][1];
            r1.z = board[y][2];
            r1.w = board[y][3];
            r2.x = board[y][4];
            r2.y = board[y][5];
            r2.z = board[y][6];
            r2.w = board[y][7];
            output[index]     = r1;
            output[index + 1] = r2;
        }
    }
}