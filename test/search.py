import subprocess

methods="static dynamic guided".split(" ")
strides="1 2 4 8 16 32 64 128".split(" ")

for method in methods:
  for stride in strides:
    time = 0.0
    for i in range(10):
      output = str(subprocess.run(["make", "run", "OMP_METHOD={}".format(method), "OMP_STRIDE={}".format(stride)], stderr=subprocess.PIPE, stdout=subprocess.DEVNULL).stderr, encoding="ascii")
      result = float(output.split("\n")[-2].split(" ")[-1])
      time += result
    time /= 10
    print("method={} stride={} time={}".format(method, stride, time))