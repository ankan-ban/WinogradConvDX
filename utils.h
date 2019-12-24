#pragma once
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <omp.h>
#include <thread>

constexpr double MAX_ERROR = 0.01;
constexpr int NUM_PRINTS = 20;

int divUp(int a, int b)
{
    return (a + b - 1) / b;
}

uint16_t Fp32ToFp16(float f32) {
    uint32_t f = *(uint32_t*)& f32;

    uint16_t f16 = 0;

    f16 |= (f >> 16) & 0x8000; // copy sign bit

    uint32_t e = (f >> 23) & 0xff; // extract exponent
    uint32_t m = f & 0x7fffff; // extract mantissa

    if (e == 255) {
        // dealing with a special here
        if (m == 0) {
            // infinity
            return (f16 | 0x7c00); // e=31, m=0, preserve sign
        }
        else {
            // NaN
            return 0x7e00; // e=31, m=0x200, s=0
        }
    }
    else if ((e >= 143) || ((e == 142) && (m > 0x7fe000))) {
        // not representable in FP16, so return infinity
        return (f16 | 0x7c00); // e=31, m=0, preserve sign
    }
    else if ((e <= 101) || ((e == 102) && (m < 0x2000))) {
        // underflow to 0
        return f16;
    }
    else if (e <= 112) {
        // denorm situation
        m |= 0x800000; // add leading 1

                       // the 24-bit mantissa needs to shift 14 bits over to
                       // fit into 10 bits, and then as many bits as the exponent
                       // is below our denorm exponent
                       //  127 (fp32 bias) 
                       // -  e (actual fp32 exponent)
                       // + 24 (fp32 mantissa bits including leading 1)
                       // - 10 (fp16 mantissa bits not including leading 1)
                       // - 15 (fp16 denorm exponent)
                       // = 126 - e
        m >>= (126 - e);

        return (uint16_t)(f16 | m); // e=0, preserve sign
    }
    else {
        // can convert directly to fp16
        e -= 112; // 127 - 15 exponent bias
        m >>= 13; // 23 - 10 mantissa bits
        return (uint16_t)(f16 | (e << 10) | m);
    }
}

float Fp16ToFp32(uint16_t f16) {
    uint32_t f = f16;

    uint32_t f32 = 0;

    f32 |= (f << 16) & 0x80000000; // copy sign bit

    uint32_t e = (f >> 10) & 0x1f; // extract exponent
    uint32_t m = f & 0x3ff; // extract mantissa

    if (e == 0) {
        if (m == 0) {
            // nothing to do; it's already +/- 0
        }
        else {
            // denorm
            e = 113;
            m <<= 13;
            // shift mantissa until the top bit is 1<<23
            // note that we've alrady guaranteed that the
            // mantissa is non-zero and that the top bit is 
            // at or below 1<<23
            while (!(m & 0x800000)) {
                e--;
                m <<= 1;
            }
            m &= 0x7fffff;

            f32 |= (e << 23) | m;
        }
    }
    else if (e == 31) {
        // FP special
        if (m == 0) {
            // Inf
            f32 |= 0x7f800000; // e=255, m=0, preserve sign
        }
        else {
            // NaN
            f32 = 0x7fc00000; // e=255, m=0x800000, s=0
        }
    }
    else {
        e += 112; // 127-15 exponent bias
        m <<= 13; // 23-10 mantissa bits
        f32 |= (e << 23) | m;
    }

    return *(float*)& f32;
}

void compareResults(void *arr1, void *arr2, int size, bool testFp16)
{
    double maxError = 0;
    double totalError = 0;
    float max_err_a = 0, max_err_b = 0;
    int max_err_index = 0;
    printf("\nFirst few elements: ");

    int numPrints = 0;
    int nanCount = 0;

    for (int i = 0; i < size; i++)
    {
        float a, b;
        if (testFp16)
        {
            a = Fp16ToFp32(((uint16_t*)arr1)[i]);
            b = Fp16ToFp32(((uint16_t*)arr2)[i]);
        }
        else
        {
            a = ((float*)arr1)[i];
            b = ((float*)arr2)[i];
        }

        float error = fabs(a - b);
        float bigger = fabs(std::fmax(a, b));
        double percentError = error;
        if (bigger)
            percentError /= bigger;

#if 0
        if (i < 20)
        {
            printf("\n%04d:  %12.8f, %12.8f, .... %11.8f", i, a, b, percentError*100);
        }
#else
        if (percentError > MAX_ERROR && numPrints < NUM_PRINTS)
        {
            printf("\n%04d:  %12.8f, %12.8f, .... %11.8f", i, a, b, percentError * 100);
            numPrints++;
        }
#endif

        if (percentError > maxError)
        {
            maxError = percentError;
            max_err_a = a;
            max_err_b = b;
            max_err_index = i;
        }

        if (percentError == percentError)   // NaN check!
            totalError += percentError;
        else
            nanCount++;
    }

    double avgError = totalError / size;
    avgError *= 100;
    maxError *= 100;

    printf("\nMax error: %f, avg error: %f, max error pair:", maxError, avgError);
    printf("\n%04d:  %12.8f, %12.8f\n", max_err_index, max_err_a, max_err_b);

    if (nanCount)   // generally bad!
        printf("\n***NaN count: %d***\n", nanCount);
}

void fillRandomArray(void *out, size_t size, bool testFp16)
{
    // fill between 0 and 1
    if (testFp16)
    {
        uint16_t *arr = (uint16_t*)out;

        for (size_t i = 0; i < size; i++)
        {
            arr[i] = Fp32ToFp16(((float)(rand())) / RAND_MAX);
        }
    }
    else
    {
        float *arr = (float *)out;

        for (int i = 0; i < size; i++)
        {
            arr[i] = ((float)(rand())) / RAND_MAX;
        }
    }
}

void matrixMulCPU(int M, int N, int K, int batch, void* c, void* a, void* b, bool fp16 = true)
{
    float *fA, *fB;
    if (fp16)
    {
        fA = (float*) malloc(M*K*batch*sizeof(float));
        fB = (float*) malloc(K*N*batch*sizeof(float));

        for (int i = 0; i < M*K*batch; i++)
            fA[i] = Fp16ToFp32(((uint16_t*)a)[i]);

        for (int i = 0; i < K*N*batch; i++)
            fB[i] = Fp16ToFp32(((uint16_t*)b)[i]);
    }
    else
    {
        fA = (float*)a;
        fB = (float*)b;
    }
    for (int b = 0; b < batch; b++)
    {
        for (int i = 0; i < M; ++i)
            for (int j = 0; j < N; ++j)
            {
                float S = 0;
                for (int k = 0; k < K; ++k)
                    S += fA[b*M*K+ i * K + k] * fB[b*K*N + k * N + j];
                if (fp16)
                    ((uint16_t*)c)[b*M*N + i * N + j] = Fp32ToFp16(S);
                else
                    ((float*)c)[b*M*N + i * N + j] = S;
            }
    }

    if (fp16)
    {
        free(fA);
        free(fB);
    }
}

template<int M, int N, int K, typename T>
void matrixMulCPU(T *c, T *a, T *b)
{
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j)
        {
            float S = 0;
            for (int k = 0; k < K; ++k)
                S += (float)(a[i*K + k]) * (float)(b[k*N + j]);
            c[i*N + j] = (T)S;
        }
}

template<typename T>
void filterTransform4x4(T *transformedFilter, T *filter)
{
    // transform applied to filter (of size 3x3)
    T G[6 * 3] = {
        1.0/4 ,      0 ,      0 ,
       -1.0/6 , -1.0/6 , -1.0/6 ,
       -1.0/6 ,  1.0/6 , -1.0/6 ,
        1.0/24,  1.0/12,  1.0/6 ,
        1.0/24, -1.0/12,  1.0/6 ,
        0     ,      0 ,      1
    };

    T Gt[3 * 6] = {
        1.0/4,    -1.0/6,     -1.0/6,     1.0/24,     1.0/24,   0,
        0,        -1.0/6,      1.0/6,     1.0/12,    -1.0/12,   0,
        0,        -1.0/6,     -1.0/6,     1.0/6,      1.0/6,    1
    };

    T tempFilter[6 * 3];
    matrixMulCPU<6, 3, 3, T>(tempFilter, G, filter);
    matrixMulCPU<6, 6, 3, T>(transformedFilter, tempFilter, Gt);
}


#define INDEX_NCHW(n,c,h,w) ((n)*C*H*W + (c)*H*W + (h)*W + w)
#define INDEX_NKHW(n,c,h,w) ((n)*K*H*W + (c)*H*W + (h)*W + w)
#define FILTER_IDX_NCHW(k,c,h,w) ((k)*C*S*R + (c)*S*R + (h)*R + w)

// transform KCSR (256x256x3x3) filter for winograd
// output is 6x6x256x256
void transformFilterTensor_Winograd4x4(int K, int C, void *transformedFilter, const void *weight, bool fp16)
{
    constexpr int S = 3;
    constexpr int R = 3;

    for (int k = 0; k < K; k++)
    {
        for (int c = 0; c < C; c++)
        {
            // 1. read single filter from memory
            float filterTile[3][3];
            for (int s = 0; s < S; s++)
                for (int r = 0; r < R; r++)
                {
                    filterTile[s][r] = fp16 ? Fp16ToFp32(((uint16_t*)weight)[FILTER_IDX_NCHW(k, c, s, r)]) : 
                        ((float*)weight)[FILTER_IDX_NCHW(k, c, s, r)];
                }

            // 2. transform it
            float transformedFilterTile[6][6];
            filterTransform4x4(&(transformedFilterTile[0][0]), &(filterTile[0][0]));

            // 3. write it back to memory (in HWCK layout)
            for (int i = 0; i < 6; i++)
                for (int j = 0; j < 6; j++)
                {
                    if (fp16)
                        ((uint16_t*)transformedFilter)[i * 6 * C*K + j * C*K + c * K + k] = Fp32ToFp16(transformedFilterTile[i][j]);
                    else
                        ((float*)transformedFilter)[i * 6 * C*K + j * C*K + c * K + k] = transformedFilterTile[i][j];
                }
        }
    }
}


// cpu implementation of convolution
// (convolution with zero padding)
// N - mini-batch size
// K - num of output channels
// C - num of input channels
// H - height
// W - width
// S - height of filter
// R - width of filter
void convRef(int N, int K, int C, int H, int W, int S, int R, bool fp16, 
             void *output, const void *input, const void *weight, const void *bias, bool relu)
{
    int threads = std::thread::hardware_concurrency();
    //printf("\nthreads: %d\n", threads);
    omp_set_num_threads(threads);

    #pragma omp parallel for
    for (int n = 0; n < N; n++)
    {
        for (int k = 0; k < K; k++)
        {
            for (int h = 0; h < H; h++)
            {
                for (int w = 0; w < W; w++)
                {
                    float op = 0.0f;
                    for (int c = 0; c < C; c++)
                    {
                        for (int s = 0; s < S; s++)
                        {
                            for (int r = 0; r < R; r++)
                            {
                                float filter = fp16 ? Fp16ToFp32(((uint16_t*)weight)[FILTER_IDX_NCHW(k, c, s, r)]) : 
                                                      (((float*)weight)[FILTER_IDX_NCHW(k, c, s, r)]);
                                int y = h + s - S / 2;
                                int x = w + r - R / 2;
                                float ip = 0;
                                if (y >= 0 && y < H && x >= 0 && x < W)
                                    ip = fp16 ? Fp16ToFp32(((uint16_t*)input)[INDEX_NCHW(n, c, y, x)]) :
                                                (((float*)input)[INDEX_NCHW(n, c, y, x)]);
                                op += ip * filter;
                            }   // r
                        }   // s
                    }   // c

                    if (bias)
                        op += fp16 ? Fp16ToFp32(((uint16_t*)bias)[k]) : ((float*)bias)[k];

                    if (relu && op < 0)
                        op = 0;

                    if (fp16)
                    {
                        ((uint16_t*)output)[INDEX_NKHW(n, k, h, w)] = Fp32ToFp16(op);
                    }
                    else
                    {
                        ((float*)output)[INDEX_NKHW(n, k, h, w)] = op;
                    }

                }   // w
            } // h
        } // k
    } // n
}
