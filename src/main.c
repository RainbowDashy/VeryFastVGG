#include <stdio.h>
#include <time.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include "matrix.h"

void help() {
    const char *help_str =
        "Usage:\n"
        "\tvgg11_bn <weights_path> <image_path> <output_file>\n";
    printf("%s", help_str);
}

void* mapFile(const char *path) {
    int fd = open(path, O_RDONLY);
    struct stat sb;
    fstat(fd, &sb);
    void *mem = mmap(NULL, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    return mem;
}

void solve(const char **argv) {
    void *weightMem = mapFile(argv[1]);
    void *inputMem = mapFile(argv[2]);

    Matrix *input = mnew(1), *weight = mnew(0), *bias = mnew(0), *output = mnew(1);
    Matrix *mean = mnew(0), *var = mnew(0);
    minit(input, 1, 3, 224, 224, &inputMem);

    int feature[13] = {64, -1, 128, -1, 256, 256, -1, 512, 512, -1, 512, 512, -1};
    for (int i = 0; i < 13; ++i) {
        if (feature[i] == -1) {
            MaxPool2d(input, output);
            mswap(&input, &output);
        } else {
            minit(weight, feature[i], input->b, 3, 3, &weightMem);
            minit(bias, 1, 1, 1, feature[i], &weightMem);
            Conv2d(input, weight, bias, output);
            mswap(&input, &output);
            minit(weight, 1, 1, 1, input->b, &weightMem);
            minit(bias, 1, 1, 1, input->b, &weightMem);
            minit(mean, 1, 1, 1, input->b, &weightMem);
            minit(var, 1, 1, 1, input->b, &weightMem);
            BatchNorm2d(input, weight, bias, mean, var, output);
            mswap(&input, &output);
            ReLUInplace(input);
        }
    }
    AdaptiveAvgPool2d(input, output);
    mswap(&input, &output);

    // flatten
    input->d = msize(input);
    input->a = 1;
    input->b = 1;
    input->c = 1;

    minit(weight, 1, 1, 4096, 512 * 7 * 7, &weightMem);
    minit(bias, 1, 1, 1, 4096, &weightMem);
    Linear(input, weight, bias, output);
    mswap(&input, &output);

    ReLUInplace(input);

    DropoutInplace(input);

    minit(weight, 1, 1, 4096, 4096, &weightMem);
    minit(bias, 1, 1, 1, 4096, &weightMem);
    Linear(input, weight, bias, output);
    mswap(&input, &output);

    ReLUInplace(input);

    DropoutInplace(input);

    minit(weight, 1, 1, 1000, 4096, &weightMem);
    minit(bias, 1, 1, 1, 4096, &weightMem);
    Linear(input, weight, bias, output);

    FILE *outputFile = fopen(argv[3], "w");
    for (int i = 0; i < msize(output); ++i)
        fprintf(outputFile, "%.8f\n", output->v[i]);
    fclose(outputFile);
}

int main(int argc, char const *argv[]) {
    if (argc != 4) {
        help();
        return 0;
    }
    omp_set_num_threads(4);
    double start = omp_get_wtime(), diff;
    solve(argv);
    diff = omp_get_wtime() - start;
    fprintf(stderr, "Total time: %lf\n", diff);
    return 0;
}
