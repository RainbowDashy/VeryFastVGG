#pragma once
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <string.h>
#include <immintrin.h>

#ifndef OMP_METHOD
#define OMP_METHOD static
#endif
#ifndef OMP_STRIDE
#define OMP_STRIDE 1
#endif

typedef float db;

double clk;
void enter() {
    clk = omp_get_wtime();
}
void leave(const char *msg) {
    double t = omp_get_wtime() - clk;
    fprintf(stderr, "%s\nElapsed time: %lf\n", msg, t);
}

typedef struct {
    int a, b, c, d;
    int allo;
    db *v;
} Matrix;

void mshape(Matrix *m, int a, int b, int c, int d) {
    m->a = a;
    m->b = b;
    m->c = c;
    m->d = d;
}

#define msize(m) \
    (m->a * m->b * m->c * m->d)

#define mpos(m, i, j, k, l) \
    ((i) * m->b * m->c * m->d + (j) * m->c * m->d + (k) * m->d + (l))

#define mget(m, i, j, k, l) \
    (m->v[mpos(m, i, j, k, l)])

void mset(Matrix *m, int i, int j, int k, int l, db v) {
    m->v[mpos(m, i, j, k, l)] = v;
}

void mswap(Matrix **a, Matrix **b) {
    Matrix *c = *a;
    *a = *b;
    *b = c;
}

void mallo(Matrix *m) {
    if (!m->allo)
        return;
    if (m->v != NULL) {
        free(m->v);
    }
    m->v = (db *)malloc(sizeof(db) * m->a * m->b * m->c * m->d);
}

void mread(Matrix *m, void **mem) {
    int bytes = msize(m) * sizeof(db);
    if (m->allo)
        memcpy(m->v, *mem, bytes);
    else
        m->v = (db *) (*mem);
    *mem += bytes;
}

void minit(Matrix *m, int i, int j, int k, int l, void **mem) {
    mshape(m, i, j, k, l);
    mallo(m);
    mread(m, mem);
}

Matrix *mnew(int allo) {
    Matrix *ret = malloc(sizeof(Matrix));
    ret->v = NULL;
    ret->allo = allo;
    return ret;
}

void mprint(Matrix *m) {
    printf("\n\n%d %d %d %d\n", m->a, m->b, m->c, m->d);
    printf("[");
    for (int i = 0; i < m->a; ++i) {
        printf("[");
        for (int j = 0; j < m->b; ++j) {
            printf("[");
            for (int k = 0; k < m->c; ++k) {
                printf("[");
                for (int l = 0; l < m->d; ++l) {
                    printf("%.2f ", mget(m, i, j, k, l));
                }
                printf("]\n ");
            }
            printf("]\n");
        }
        printf("] ");
    }
    printf("]\n");
}

void ReLU(Matrix *input, Matrix *output) {
    enter();
    mshape(output, input->a, input->b, input->c, input->d);
    mallo(output);
    for (int i = 0; i < msize(input); ++i) {
        output->v[i] = input->v[i] > 0 ? input->v[i] : 0;
    }
    leave("ReLU");
}

void ReLUInplace(Matrix *input) {
    enter();
    for (int i = 0; i < msize(input); ++i) {
        input->v[i] = input->v[i] > 0 ? input->v[i] : 0;
    }
    leave("ReLUInplace");
}

void Dropout(Matrix *input, Matrix *output) {
    return ;
    mshape(output, input->a, input->b, input->c, input->d);
    mallo(output);
    for (int i = 0; i < msize(input); ++i) {
        output->v[i] = (rand() & 1) ? 0 : input->v[i];
    }
}

void DropoutInplace(Matrix *input) {
    return ;
    for (int i = 0; i < msize(input); ++i) {
        if (rand() & 1) input->v[i] = 0;
    }
}

db mmean(db *v, int size) {
    db res = 0;
    for (int i = 0; i < size; ++i) res += v[i];
    res /= size;
    return res;
}

db mvar(db *v, int size, db mean) {
    db res = 0;
    for (int i = 0; i < size; ++i) {
        res += (v[i] - mean) * (v[i] - mean);
    }
    if (size > 1) res /= size - 1;
    return res;
}

void BatchNorm2d(Matrix *input, Matrix *weight, Matrix *bias, Matrix *mean,
                 Matrix *var, Matrix *output) {
    enter();
    mshape(output, input->a, input->b, input->c, input->d);
    mallo(output);
    db eps = 1e-5;
    // this should only work when input->a = 1
    for (int i = 0, idx = 0; i < msize(input);
         i += input->c * input->d, ++idx) {
        // loop invariant
        db sq = sqrt(var->v[idx] + eps);
        for (int j = i; j < i + input->c * input->d; ++j) {
            output->v[j] = (input->v[j] - mean->v[idx]) /
                               sq * weight->v[idx] +
                           bias->v[idx];
        }
    }
    leave("BatchNorm2d");
}

void Linear(Matrix *input, Matrix *weight, Matrix *bias, Matrix *output) {
    enter();
    mshape(output, input->a, input->b, input->c, weight->c);
    mallo(output);
    #pragma omp parallel shared(input, weight, bias, output) num_threads(4)
    #pragma omp for schedule(OMP_METHOD, OMP_STRIDE)
    for (int i = 0; i < weight->c; ++i) {
        output->v[i] = bias->v[i];
        for (int j = 0; j < weight->d; ++j)
            output->v[i] += weight->v[i * weight->d + j] * input->v[j];
    }
    leave("Linear");
}

db max2(db a, db b) { return a > b ? a : b; }

db max4(db a, db b, db c, db d) { return max2(a, max2(b, max2(c, d))); }

// kernel_size=2 stride=2
void MaxPool2d(Matrix *input, Matrix *output) {
    enter();
    mshape(output, input->a, input->b, input->c / 2, input->d / 2);
    mallo(output);
    int tot = 0;
    for (int s = 0; s < msize(input); s += input->c * input->d) {
        for (int ii = 0; ii < input->c; ii += 2) {
            for (int jj = 0; jj < input->d; jj += 2) {
                output->v[tot++] =
                    max4(input->v[s + ii * input->d + jj],
                         input->v[s + ii * input->d + jj + 1],
                         input->v[s + (ii + 1) * input->d + jj],
                         input->v[s + (ii + 1) * input->d + jj + 1]);
            }
        }
    }
    leave("MaxPool2d");
}

// output_size = (7, 7)
// NOT adaptive!
void AdaptiveAvgPool2d(Matrix *input, Matrix *output) {
    enter();
    mshape(output, input->a, input->b, 7, 7);
    mallo(output);
    int tot = 0;
    int size = input->c / 7;
    for (int s = 0; s < msize(input); s += input->c * input->d) {
        for (int ii = 0; ii < input->c; ii += size) {
            for (int jj = 0; jj < input->d; jj += size) {
                db avg = 0;
                for (int i = ii; i < ii + size; ++i) {
                    for (int j = jj; j < jj + size; ++j) {
                        avg += input->v[s + ii * input->d + jj];
                    }
                }
                output->v[tot++] = avg / (size * size);
            }
        }
    }
    leave("AdaptiveAvgPool2d");
}

void Padding(Matrix *input, Matrix *output) {
    enter();
    mshape(output, input->a, input->b, input->c + 2, input->d + 2);
    mallo(output);
    for (int i = 0; i < output->a; ++i)
        for (int j = 0; j < output->b; ++j)
            for (int k = 0; k < output->c; ++k)
                for (int l = 0; l < output->d; ++l) {
                    if (k == 0 || k == output->c - 1 ||
                        l == 0 || l == output->d - 1)
                        mset(output, i, j, k, l, 0);
                    else
                        mset(output, i, j, k, l, mget(input, i, j, k - 1, l - 1));
                }
    leave("Padding");
}

// a b c d -> a c d b
void Transpose(Matrix *input, Matrix *output) {
    enter();
    mshape(output, input->a, input->c, input->d, input->b);
    mallo(output);
    for (int i = 0; i < output->a; ++i)
        for (int j = 0; j < output->b; ++j)
            for (int k = 0; k < output->c; ++k)
                for (int l = 0; l < output->d; ++l) {
                    mset(output, i, j, k, l, mget(input, i, l, j, k));
                }
    leave("Transpose");
}

// stride = 1, kernel = 3 * 3
void Conv2d(Matrix *input, Matrix *weight, Matrix *bias, Matrix *output) {
    // orginal input c and input d
    int originalInputC = input->c;
    int originalInputD = input->d;
    // orignal weight a
    int originalWeightA = weight->a;

    // padding
    Padding(input, output);
    mswap(&input, &output);
    // transpose input
    Transpose(input, output);
    mswap(&input, &output);
    // transpose weight
    Matrix *newWeight = mnew(1);
    Transpose(weight, newWeight);

    // conv2d
    enter();
    mshape(output, 1, originalWeightA, originalInputC, originalInputD);
    mallo(output);
    __m256 avxSum, a, b;
    #pragma omp parallel shared(input, newWeight, bias, output) private(a, b, avxSum) num_threads(4)
    #pragma omp for schedule(OMP_METHOD, OMP_STRIDE)
    // oa = 0
    for (int ob = 0; ob < output->b; ++ob)
        for (int oc = 0; oc < output->c; ++oc)
            for (int od = 0; od < output->d; ++od) {
                db sum = bias->v[ob];
                avxSum =  _mm256_setzero_ps();
                int wd;
                // kernel 3 * 3
                #define _unroll(wb, wc) ({ \
                    wd = 0; \
                    if (newWeight->d >= 16)  \
                        for (; wd < newWeight->d; wd += 16) { \
                            db *inputP = input->v + mpos(input, 0, oc + wb, od + wc, wd); \
                            db *weightP = newWeight->v + mpos(newWeight, ob, wb, wc, wd); \
                            a = _mm256_loadu_ps(inputP); \
                            b = _mm256_loadu_ps(weightP); \
                            avxSum = _mm256_fmadd_ps(a, b, avxSum); \
                            a = _mm256_loadu_ps(inputP + 8); \
                            b = _mm256_loadu_ps(weightP + 8); \
                            avxSum = _mm256_fmadd_ps(a, b, avxSum); \
                        } \
                    else /*only one case in data, branch prediction can handle this*/ \
                        for (; wd < newWeight->d; ++wd) \
                            sum += mget(input, 0, oc + wb, od + wc, wd) * \
                                    mget(newWeight, ob, wb, wc, wd); \
                })
                _unroll(0, 0);
                _unroll(0, 1);
                _unroll(0, 2);
                _unroll(1, 0);
                _unroll(1, 1);
                _unroll(1, 2);
                _unroll(2, 0);
                _unroll(2, 1);
                _unroll(2, 2);
                avxSum = _mm256_hadd_ps(avxSum, avxSum);
                avxSum = _mm256_hadd_ps(avxSum, avxSum);
                sum += avxSum[0] + avxSum[4];
                mset(output, 0, ob, oc, od, sum);
            }

    // free memory
    free(newWeight->v);
    free(newWeight);
    leave("Conv2d");
}
