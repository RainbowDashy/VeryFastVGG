#include <stdio.h>
#include "matrix.h"

void help() {
  const char *help_str = 
    "Usage:\n"
    "\tvgg11_bn <weights_path> <image_path> <output_file>\n";
  printf(help_str);
}

int main(int argc, char const *argv[]) {
  if (argc != 4) {
    help();
  }
  return 0;
}
