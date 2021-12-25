cmake_minimum_required(VERSION 3.10)
project(VeryFastVGG)

find_package(OpenMP REQUIRED)

add_executable(vgg11_bn src/main.c)
target_compile_definitions(vgg11_bn PUBLIC OMP_METHOD=${OMP_METHOD} OMP_STRIDE=${OMP_STRIDE})
target_compile_options(vgg11_bn PRIVATE -Wall -march=native)
target_link_libraries(vgg11_bn PRIVATE OpenMP::OpenMP_C m)
