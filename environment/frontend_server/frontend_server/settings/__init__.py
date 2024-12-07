# FOR PUSHING STATIC TO AWS


# from .base import *
# from .production import *

# try:
#   from .local import *
# except:
# pass


# FOR GENERAL USES
from __future__ import annotations

from .base import *


try:
    from .local import *

    live = False
except:
    live = True

if live:
    from .production import *
