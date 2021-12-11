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

    Matrix input;
    mshape(&input, 1, 3, 224, 224);
    mcreate(&input);
    mread(&input, inputFD);
    mprint(&input);
}

int main(int argc, char const *argv[]) {
    if (argc != 4) {
        help();
        return 0;
    }
    solve(argv);
    return 0;
}
