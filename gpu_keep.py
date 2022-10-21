import time
import torch
import argparse


def benchmark(device):
    print('显卡(cuda:0)性能测试: cuda 11.7 下 3090Ti=85(理论160), V100 16GB=86(理论125), A100 80GB=231(理论312); cuda 11.4 下 2080Ti=58')
    import torch
    from torch.utils import benchmark
    typ = torch.float16
    n = 1024 * 16
    a = torch.randn(n, n).type(typ).to(device)
    b = torch.randn(n, n).type(typ).to(device)
    t = benchmark.Timer(stmt='a @ b', globals={'a': a, 'b': b})
    x = t.timeit(50)
    print(2*n**3 / x.median / 1e12)

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='..'
    )
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    # devices = [torch.device("cuda:3"), torch.device("cuda:4")]
    while 1:
        benchmark(torch.device("cuda:{}".format(args.device)))
        # time.sleep(100000)
    # benchmark(torch.device("cuda"))

    