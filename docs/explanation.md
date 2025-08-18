# Explanation

TODO:
1. Add theoretical basis
2. Add citations.

This part of the documentation contains explanations for the physical basis of things and the terminology used, with
references to articles and literature.

## Background

This project arose from a simple question I asked myself while working on my master's thesis in applied physics. The
question was "can you actually measure light intensity from photographs?", as I was researching the optical properties of
a carbon nanotube deposition. Originally this codebase was but a simple (and messy) set of scripts. Over time as I added
features to my scripts, I found myself annoyed at how messy the larger and larger set of scripts had become. Following
that I decided to convert it into a proper project. The first version that I made available publicly is found in my
other [repository](https://github.com/samivout/camera_linearity). This project was built mainly on NumPy, CuPy and SciPy,
and I was aiming to finish it into a proper release on PyPi, but as can be seen now, the repository is archived.

At the start of 2025 I started studying machine learning and deep learning, getting introduced to PyTorch. After starting
to become familiar with the package, I realized that PyTorch was perfect for this project. With PyTorch, I could avoid the
complexity  of juggling NumPy and CuPy in tandem, I could build more complex models over time with ease, and with the
autograd system I could handle the uncertainty propagation in a much more robust, less error-prone way due to not having
to compute  derivatives and implement them in code manually. This led me to rewrite the project, basing it upon PyTorch.

In terms of science and tech this project doesn't contain any new discoveries, but instead is aimed at making the
measurement of a camera's linearity and the merging of HDR images easy. The most unique aspect of this project is likely
the inclusion of uncertainty propagation computations, allowing the user to evaluate the accuracy of the results when
they provide estimates for the uncertainty of the source data. Generally, this is a continuous learning and passion
project of mine.

