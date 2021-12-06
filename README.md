# VeryFastVGG

It's faster than me!

## Build

Make sure you have `gcc`, `make`, `cmake` and `openmp` installed.

Use commands as follows, and the binary file `vgg11_bn` will be in the directory `build/`.

```shell
make
```

Some other useful commands are
- building a release type binary
```shell
make CMAKE_BUILD_TYPE=Release
```
- removing the directory `build/`
```shell
make clean
```

## TODO
- [ ] Input & Output
- [ ] nn.Conv2d
- [ ] nn.BatchNorm2d
- [ ] nn.ReLU
- [ ] nn.MaxPool2d
- [ ] nn.Linear
- [ ] nn.Dropout
- [ ] nn.AdaptiveAvgPool2d
- [ ] torch.Flaten

## Reference

https://notes.sjtu.edu.cn/s/xegV8P0H-

https://pytorch.org/docs/stable/nn.html
