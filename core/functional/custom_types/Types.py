from typing import Literal

Neuro_extension = Literal[
    '.pth',
    '.pt',
    '.pwf'
]
"""
Torches file extensions, that can be loaded.
"""

Neuro_type = Literal[
    'Conv2d',
    'Linear',
    'Dense'
]
"""
Types of neuro layers, that can be constructed.
"""
