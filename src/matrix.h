#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef long double db;

typedef struct {
    int a, b, c, d;
    db *v;
} Matrix;

int msize(Matrix *m) {
    return m->a * m->b * m->c * m->d;
}

int mpos(Matrix *m, int i, int j, int k, int l) {
    return i * m->b * m->c * m->d + j * m->c * m->d + k * m-> d + l;
}

db mget(Matrix *m, int i, int j, int k, int l) {
    return m->v[mpos(m, i, j, k, l)];
}

void mcreate(Matrix *m) {
    m->v = (db*) malloc(sizeof(db) * m->a * m->b * m->c * m->d);
}

void mread(Matrix *m, FILE *fd) {
    for (int i = 0; i < msize(m); ++i)
        fscanf(fd, "%Lf", &m->v[i]);
}

void ReLU(Matrix *input, Matrix *output) {
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
    for (int i = 0; i < msize(input); ++i) {
        output->v[i] = (rand() & 1) ? 0 : input->v[i];
    }
}

void DropoutInplace(Matrix *input) {
    for (int i = 0; i < msize(input); ++i) {
        if (rand() & 1)
            input->v[i] = 0;
    }
}

db mmean(db *v, int size) {
    db res = 0;
    for (int i = 0; i < size; ++i)
        res += v[i];
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

void BatchNorm2d(Matrix *input, Matrix *output) {
    db eps = 1e-5;
    for (int i = 0; i < msize(input); i += input->c * input->d) {
        db mean = mmean(input->v + i, input->c * input->d);
        db var = mvar(input->v + i, input->c * input->d, mean);
        for (int j = i; j < i + input->c * input->d; ++j) {
            output->v[j] = (input->v[j]-mean) / (sqrt(var) + eps);
        }
    }
}

void Linear(Matrix *input, Matrix *weight, Matrix *bias, Matrix *output) {
    for (int i = 0; i < weight->c; ++i) {
        output->v[i] = bias->v[i];
        for (int j = 0; j < weight->d; ++j)
            output->v[i] += weight->v[i * weight->c + j] * input->v[j];
    }
}

db max2(db a, db b) {
    return a > b ? a : b;
}

db max4(db a, db b, db c, db d) {
    return max2(a, max2(b, max2(c, d)));
}

// kernel_size=2 stride=2
void MaxPool2d(Matrix *input, Matrix *output) {
    int tot = 0;
    for (int s = 0; s < msize(input); s += input->c * input->d) {
        for (int ii = 0; ii < input->c; ii += 2) {
            for (int jj = 0; jj < input->d; jj += 2) {
                output->v[tot++] = max4(input->v[s + ii*input->d + jj], 
                                        input->v[s + ii*input->d + jj + 1],
                                        input->v[s + (ii+1)*input->d + jj],
                                        input->v[s + (ii+1)*input->d + jj + 1]);

            }
        }
    }
}
