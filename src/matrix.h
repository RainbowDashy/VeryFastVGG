#pragma once
#include <stdio.h>
#include <stdlib.h>

typedef long double db;

typedef struct {
    int a, b, c, d;
    db *v;
} Matrix4D;

int msize(Matrix4D *m) {
    return m->a * m->b * m->c * m->d;
}

int mpos(Matrix4D *m, int i, int j, int k, int l) {
    return i * m->b * m->c * m->d + j * m->c * m->d + k * m-> d + l;
}

db mget(Matrix4D *m, int i, int j, int k, int l) {
    return m->v[mpos(m, i, j, k, l)];
}

void mcreate(Matrix4D *m) {
    m->v = (db*) malloc(sizeof(db) * m->a * m->b * m->c * m->d);
}

void mread(Matrix4D *m, FILE *fd) {
    for (int i = 0; i < msize(m); ++i)
        fscanf(fd, "%Lf", &m->v[i]);
}

void ReLU(Matrix4D *input, Matrix4D *output) {
    for (int i = 0; i < msize(input); ++i) {
        output->v[i] = input->v[i] > 0 ? input->v[i] : 0;
    }
}

void ReLUInplace(Matrix4D *input) {
    for (int i = 0; i < msize(input); ++i) {
        input->v[i] = input->v[i] > 0 ? input->v[i] : 0;
    }
}