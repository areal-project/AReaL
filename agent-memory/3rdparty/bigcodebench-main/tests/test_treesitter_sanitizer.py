from bigcodebench.sanitize import code_extract, sanitize


def test_code_extract():
    test_simple = r"""Here is some python code generated
import numpy as np
Sorry, I made a mistake, let me try again
from numpy import sin, cos, tan

def f(x):
    return tan(x)
As you can observe from above
    """
    assert (
        code_extract(test_simple)
        == r"""from numpy import sin, cos, tan

def f(x):
    return tan(x)"""
    )

    test_empty_lines = r"""import numpy as np


import pandas
Sorry, let me try again
from numpy import sin, cos, tan
def f(x):
    return tan(x)
"""
    assert (
        code_extract(test_empty_lines)
        == r"""from numpy import sin, cos, tan
def f(x):
    return tan(x)"""
    )


def test_sanitize_simple():
    icode = r"""Following is the code snippet:
```python
import numpy as np
from numpy import sin, cos

def f(x):
    return np.tan(x)

def g(x):
    return cos(f(x))

def g(x):
    return sin(f(x))

def c(x):
    assert 1==1

assert g(0) == 1
```
"""
    assert (
        sanitize(icode)
        == r"""import numpy as np
from numpy import sin, cos
def f(x):
    return np.tan(x)
def g(x):
    return cos(f(x))"""
    )


def test_sanitize_class():
    icode = r"""Following is the code snippet:
```python
import numpy as np
from numpy import sin, cos
class g():
    def hello_world():
        return 0
def f(x):
    print(g.hello_world())
    return np.tan(x)
```
"""

    assert (
        sanitize(icode)
        == r"""import numpy as np
from numpy import sin, cos
class g():
    def hello_world():
        return 0
def f(x):
    print(g.hello_world())
    return np.tan(x)"""
    )


def test_entrypoint_basic():
    icode = r"""Following is the code snippet:
```python
import numpy as np
from numpy import sin, cos

def f(x):
    return np.tan(x)

def g(x):
    return cos(f(x))

def g(x):
    return sin(f(x))

def c(x):
    return 0

assert g(0) == 1
```
"""
    assert (
        sanitize(icode, "g")
        == r"""import numpy as np
from numpy import sin, cos
def f(x):
    return np.tan(x)
def g(x):
    return cos(f(x))"""
    )


def test_entrypoint_chain():
    icode = r"""Following is the code snippet:
```python
import numpy as np
from numpy import sin, cos

def f(x):
    return c(x)
assert f(1) == 5
def g(x):
    return cos(f(x))

def c(x):
    newObj = h()
    return x

class h():
    def hello_world():
        return 0

class h():
    def goodbye_world():
        return 0


assert g(0) == 1
```
"""
    print(sanitize(icode, "g"))
    assert (
        sanitize(icode, "g")
        == r"""import numpy as np
from numpy import sin, cos
def f(x):
    return c(x)
def g(x):
    return cos(f(x))
def c(x):
    newObj = h()
    return x
class h():
    def hello_world():
        return 0"""
    )


def test_entrypoint_no_chain():
    icode = r"""Following is the code snippet:
```python
import numpy as np
from numpy import sin, cos, sum

def f(x):
    return np.sum(x)
assert f(1) == 5
def g(x):
    return cos(f(x))

def c(x):
    newObj = h()
    return x

class h():
    def hello_world():
        return 0


assert g(0) == 1
```
"""
    assert (
        sanitize(icode, "g")
        == r"""import numpy as np
from numpy import sin, cos, sum
def f(x):
    return np.sum(x)
def g(x):
    return cos(f(x))"""
    )


def test_entrypoint_variable():
    icode = r"""Following is the code snippet:
```python
import numpy as np
from numpy import sin, cos

SOME_CONSTANT = 5

def f(x):
    return c(x)
assert f(1) == 5
def g(x):
    return cos(f(x))

def c(x):
    newObj = h()
    return x

class h():
    def hello_world():
        return SOME_CONSTANT

def d(x):
    return g(x)


assert g(0) == 1
```
"""

    assert (
        sanitize(icode, "g")
        == r"""import numpy as np
from numpy import sin, cos
SOME_CONSTANT = 5
def f(x):
    return c(x)
def g(x):
    return cos(f(x))
def c(x):
    newObj = h()
    return x
class h():
    def hello_world():
        return SOME_CONSTANT"""
    )
