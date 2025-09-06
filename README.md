# torch-swiglu-compat
A small compatability patch to add in fmo-mt's style of SwiGLU into PyTorch's nn while we're waiting for the actual PR to be merged. Most of the functional code is based on fmo-mt' work, so I don't make any claim to copyright over any of it. This is just a temporary solution and I'll take it down once the full SwiGLU support is added to Torch.

A tiny **drop-in shim** that exposes **`torch.nn.SwiGLU`** and **`torch.nn.functional.swiglu`**
exactly as in PyTorch PR **#144465** (Support Swiglu for Module and functional), so you can
use the **final API** today and remove this later with zero code changes.

> This package defines the symbols only if they do **not** already exist in your PyTorch.
> When upstream lands, it no-ops and your code keeps working.

## Install (with `uv`)

```bash
uv pip install git+https://github.com/Lexa-B/torch-swiglu-compat.git

```

Then in your code, install the package:

```python
import torch_swiglu_compat as swiglu_compat
swiglu_compat.install()
```
## Usage

```python
import torch_swiglu_compat as swiglu_compat
swiglu_compat.install()

import torch.nn as nn
import torch.nn.functional as F

model = nn.Sequential(
    nn.Linear(1024, 2048),
    nn.SwiGLU(dim=-1), # the input dim is twice the output dim
    nn.Linear(1024, 1024),
)
# Also works functionally:
# y = F.swiglu(x, dim=-1)
```

Contract (matches PyTorch PR #144465)

* torch.nn.functional.swiglu(input, dim=-1)
  * One tensor input; splits in half along dim.
  * Returns SiLU(first_half) * second_half.
  * Errors if dtype is non-floating or size on dim is odd.
* torch.nn.SwiGLU(dim=-1)
  * Thin module wrapper over the functional, for use in nn.Sequential.

