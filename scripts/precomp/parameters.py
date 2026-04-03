#!/usr/bin/env python3
# Adapted from the SQIsign reference implementation:
# https://github.com/SQISign/the-sqisign
#
# Original authors: the SQIsign team (Maria Corte-Real Santos, Joy Hall,
# Michael Meyer, Giacomo Pope, Damien Robert, et al.)
# Licensed under the Apache License, Version 2.0.

from sage.all import *
proof.all(False)  # faster

import re
for l in open('sqisign_parameters.txt'):
    for k in ('lvl', 'p', 'num_orders'):
        m = re.search(rf'^\s*{k}\s*=\s*([x0-9a-f]+)', l)
        if m:
            v = ZZ(m.groups()[0], 0)
            globals()[k] = v

f = (p+1).valuation(2)

__all__ = ['lvl', 'p', 'f', 'num_orders']
