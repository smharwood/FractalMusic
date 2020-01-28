# FractalMusic
Code for an artistic collaboration: how does math inspire music, and how does music influence the math that produces images?

Specifically, this is exploring how to seed the generation of a fractal created by the [Chaos Game](http://en.wikipedia.org/wiki/Chaos_game) with live music.

# Setup and Installation
On a Debian-based Linux distribution like Ubuntu, set up a virtual environment as usual:
```python3 -m virtualenv venv```
Activate environment and install requirements:
```source venv/bin/activate
pip install numpy scipy matplotlib mido sounddevice
```

Significant speed-ups can be attained if a Fortran 90 compiler is available on the system; for instance:
```sudo apt-get install gfortran```

# Usage
Run with `python ChaosMusic.py` to listen to ambient sound, analyze the frequencies, and seed the fractal generation with parameters determined from the sound somehow.
