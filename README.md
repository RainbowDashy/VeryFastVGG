# VeryFastVGG

It's faster than me!

## Build

Make sure you have `gcc`, `make`, `cmake`, `openmp` and `python` installed.

Use commands as follows, and the binary file `vgg11_bn` will be in the directory `build/`.

```shell
make
```

If you want to run the binary, you need to get & preprocess data. And the command is

```shell
make data
```

Some other useful commands are
- running `vgg11_bn` with default arguments
```shell
make run
```
- building a release type binary
```shell
make CMAKE_BUILD_TYPE=Release
```
- removing the directory `build/`
```shell
make clean
```
- comparing the result with pytorch's
```shell
make test
```

## TODO
- [x] Input & Output
- [x] nn.Conv2d
- [x] nn.BatchNorm2d
- [x] nn.ReLU
- [x] nn.MaxPool2d
- [x] nn.Linear
- [x] nn.Dropout
- [x] nn.AdaptiveAvgPool2d
- [x] torch.Flatten
- [x] Timing
- [x] Openmp
- [x] AVX
- [x] Loop unroll
- [x] Way to search optimal parameters
## Reference

https://notes.sjtu.edu.cn/s/xegV8P0H-

https://pytorch.org/docs/stable/nn.html
