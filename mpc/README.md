# MPC benchmark

```
python launcher.py --help
```

## Simulate two parties on one host
```
cd benchmarks
python launcher.py --multiprocess
```
### Default parameters
```
world size = 2
epochs = 20
start epoch = 0
batch size = 64
learning rate = 0.1
momentum = 0.99
seed = 42
loss function = cross entropy (ce)
```

### change parameters
example
```
python launcher.py --multiprocess --loss-func mse
```