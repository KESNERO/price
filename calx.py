#!/usr/bin/env python3

import sys

from scipy.stats import norm

if __name__ == "__main__":
    y = float(sys.argv[1])
    x = norm.ppf(y)
    print(x)

