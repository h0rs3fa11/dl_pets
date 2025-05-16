# MPC benchmark

```
cd benchmarks
python launcher.py --help
```
## plaintext baseline

```
python baseline.py
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

## AWS Instances

1. Deploy two aws instances (16 GB storage, xxx)
2. Install necessary packages on it(pip, crypten, requirement.txt)
3. Copy the training data to it (benchmarks/party0, benchmarks/party1)