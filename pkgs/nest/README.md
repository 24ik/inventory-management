
# Nest library

This library uses parts of [torchbeast](https://github.com/facebookresearch/torchbeast) almost as is (this sentence and the license have been added), distributed in the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).

```shell
CXX=c++ pip install . -vv
```

Usage in Python:

```python
import torch
import nest

t1 = torch.tensor(0)
t2 = torch.tensor(1)
d = {'hey': torch.tensor(2)}

print(nest.map(lambda t: t + 42, (t1, t2, d)))
# --> (tensor(42), tensor(43), {'hey': tensor(44)})
```
