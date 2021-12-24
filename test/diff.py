from os import read
import sys
A=[]
B=[]
eps=1e-5
def read_from_file(path, arr):
  file = open(path)
  for line in file.readlines():
    number = float(line)
    arr.append(number)
  file.close()

if __name__ == "__main__":
  read_from_file(sys.argv[1], A)
  read_from_file(sys.argv[2], B)
  for i in range(len(A)):
    if abs(A[i] - B[i]) >= eps:
      print("line {}: {} and {}, not passed", i + 1, A[i], B[i])
      exit(1)
  print("passed")
