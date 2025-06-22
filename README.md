# disk
Library for profiling and strength and resonance calculation of turbomachine disk.

## About
- strength calculation by "2-calculation" method.
- plotting the Campbell frequency diagram.
- profiling of a uniformly heated disk of equal strength loaded only by its own centrifugal force.

## Installation
```bash
pip install -r requirements.txt
# or
pip install --upgrade git+https://github.com/ParkhomenkoDV/substance.git@master
```

## Usage

```python
from disk import Disk

d = Disk(...)
d.do_smth(...)
```

See tutorial in disk/examples/

## Project structure
```text
disk/
|--- examples/  # tutorial
|--- images/  # docs images
|--- src/disk/  # source code
|--- tests
|--- .gitignore
|--- README.md  
|--- requirements.txt
|--- setup.py
```

