#pragma once

typedef long double db;

typedef struct {
    db *v;
    int a, b, c, d;
} Matrix4D;

int pos(Matrix4D *m, int i, int j, int k, int l) {
    return i * m->b * m->c * m->d + j * m->c * m->d + k * m-> d + l;
}

db get(Matrix4D *m, int i, int j, int k, int l) {
    return m->v[pos(m, i, j, k, l)];
}
