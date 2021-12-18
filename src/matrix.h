#pragma once
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

typedef long double db;

typedef struct {
    int a, b, c, d;
    db *v;
} Matrix;

void mshape(Matrix *m, int a, int b, int c, int d) {
    m->a = a;
    m->b = b;
    m->c = c;
    m->d = d;
}

int msize(Matrix *m) { return m->a * m->b * m->c * m->d; }

int mpos(Matrix *m, int i, int j, int k, int l) {
    return i * m->b * m->c * m->d + j * m->c * m->d + k * m->d + l;
}

db mget(Matrix *m, int i, int j, int k, int l) {
    return m->v[mpos(m, i, j, k, l)];
}
void mset(Matrix *m, int i, int j, int k, int l, db v) {
    m->v[mpos(m, i, j, k, l)] = v;
}

void mswap(Matrix **a, Matrix **b) {
    Matrix *c = *a;
    *a = *b;
    *b = c;
}

void mallo(Matrix *m) {
    if (m->v != NULL) {
        free(m->v);
    }
    m->v = (db *)malloc(sizeof(db) * m->a * m->b * m->c * m->d);
}

void mread(Matrix *m, FILE *fd) {
    for (int i = 0; i < msize(m); ++i) fscanf(fd, "%Lf", &m->v[i]);
}

void minit(Matrix *m, int i, int j, int k, int l, FILE *fd) {
    mshape(m, i, j, k, l);
    mallo(m);
    mread(m, fd);
}

Matrix *mnew() {
    Matrix *ret = malloc(sizeof(Matrix));
    ret->v = NULL;
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
                    printf("%.2Lf ", mget(m, i, j, k, l));
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
    mshape(output, input->a, input->b, input->c, input->d);
    mallo(output);
    for (int i = 0; i < msize(input); ++i) {
        output->v[i] = input->v[i] > 0 ? input->v[i] : 0;
    }
}

void ReLUInplace(Matrix *input) {
    for (int i = 0; i < msize(input); ++i) {
        input->v[i] = input->v[i] > 0 ? input->v[i] : 0;
    }
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
    mshape(output, input->a, input->b, input->c, input->d);
    mallo(output);
    db eps = 1e-5;
    // this should only work when input->a = 1
    for (int i = 0, idx = 0; i < msize(input);
         i += input->c * input->d, ++idx) {
        for (int j = i; j < i + input->c * input->d; ++j) {
            output->v[j] = (input->v[j] - mean->v[idx]) /
                               sqrt(var->v[idx] + eps) * weight->v[idx] +
                           bias->v[idx];
        }
    }
}

void Linear(Matrix *input, Matrix *weight, Matrix *bias, Matrix *output) {
    mshape(output, input->a, input->b, input->c, weight->c);
    mallo(output);
    for (int i = 0; i < weight->c; ++i) {
        output->v[i] = bias->v[i];
        for (int j = 0; j < weight->d; ++j)
            output->v[i] += weight->v[i * weight->d + j] * input->v[j];
    }
}

db max2(db a, db b) { return a > b ? a : b; }

db max4(db a, db b, db c, db d) { return max2(a, max2(b, max2(c, d))); }

// kernel_size=2 stride=2
void MaxPool2d(Matrix *input, Matrix *output) {
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
}

// output_size = (7, 7)
// NOT adaptive!
void AdaptiveAvgPool2d(Matrix *input, Matrix *output) {
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
}

void Conv2d(Matrix *input, Matrix *weight, Matrix *bias, Matrix *output) {
    // n
    mshape(output, input->a, weight->a, input->c, input->d);
    mallo(output);
    // the axis of output
    for (int oi = 0; oi < weight->a; ++oi)
        for (int oj = 0; oj < output->c; ++oj)
            for (int ok = 0; ok < output->d; ++ok) {
                db sum = bias->v[oi];
                // the axis of weight
                for (int ii = 0; ii < weight->b; ++ii)
                    for (int jj = 0; jj < 3; ++jj)
                        for (int kk = 0; kk < 3; ++kk) {
                            // the axis of input
                            // input[0][ii][oj + jj][ok + kk]
                            if (oj + jj == 0 || oj + jj == input->c + 1 ||
                                ok + kk == 0 || ok + kk == input->d + 1) {
                                // it's padding
                            } else {
                                sum += mget(input, 0, ii, oj + jj - 1,
                                            ok + kk - 1) *
                                       mget(weight, oi, ii, jj, kk);
                            }
                        }
                mset(output, 0, oi, oj, ok, sum);
            }
}
