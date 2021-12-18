#include <stdio.h>

#include "matrix.h"

void help() {
    const char *help_str =
        "Usage:\n"
        "\tvgg11_bn <weights_path> <image_path> <output_file>\n";
    printf("%s", help_str);
}

void solve(const char **argv) {
    FILE *weightFD = fopen(argv[1], "r");
    FILE *inputFD = fopen(argv[2], "r");
    FILE *outputFD = fopen(argv[3], "w");

    Matrix *input = mnew(), *weight = mnew(), *bias = mnew(), *output = mnew();
    Matrix *mean = mnew(), *var = mnew();
    minit(input, 1, 3, 224, 224, inputFD);

    int feature[13] = {64, -1, 128, -1, 256, 256, -1, 512, 512, -1, 512, 512, -1};
    for (int i = 0; i < 13; ++i) {
        if (feature[i] == -1) {
            MaxPool2d(input, output);
            mswap(&input, &output);
        } else {
            minit(weight, feature[i], input->b, 3, 3, weightFD);
            minit(bias, 1, 1, 1, feature[i], weightFD);
            Conv2d(input, weight, bias, output);
            mswap(&input, &output);
            minit(weight, 1, 1, 1, input->b, weightFD);
            minit(bias, 1, 1, 1, input->b, weightFD);
            minit(mean, 1, 1, 1, input->b, weightFD);
            minit(var, 1, 1, 1, input->b, weightFD);
            BatchNorm2d(input, weight, bias, output);
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

    minit(weight, 1, 1, 4096, 512 * 7 * 7, weightFD);
    minit(bias, 1, 1, 1, 4096, weightFD);
    Linear(input, weight, bias, output);
    mswap(&input, &output);

    ReLUInplace(input);

    DropoutInplace(input);

    minit(weight, 1, 1, 4096, 4096, weightFD);
    minit(bias, 1, 1, 1, 4096, weightFD);
    Linear(input, weight, bias, output);
    mswap(&input, &output);

    ReLUInplace(input);

    DropoutInplace(input);

    minit(weight, 1, 1, 1000, 4096, weightFD);
    minit(bias, 1, 1, 1, 4096, weightFD);
    Linear(input, weight, bias, output);

    mprint(output);
}

int main(int argc, char const *argv[]) {
    if (argc != 4) {
        help();
        return 0;
    }
    solve(argv);
    return 0;
}
