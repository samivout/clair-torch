# Explanation

TODO:
1. Complete theoretical basis (IN PROGRESS)
2. Add citations.

This page contains both the theoretical and a general background for this project.

## Theoretical basis

Digital camera sensors come in two types: the charge-coupled device (CCD) and the complementary metal-oxide-semiconductor
(CMOS) sensor, both based on the metal–oxide–semiconductor field-effect transistor (MOSFET)[36, 38]. They have certain
differences in have charge is quantified, but the charge generation is based on the photo electric effect in both. 

The charge accumulated in the CCD and CMOS elements is mostly linear as it is proportional to the irradiance the elements
are subjected to[36, p. 173]. Based on this one might expect that the output of the image sensor is linear. However, in
reality this is seldom the case as commercial camera’s typically apply nonlinear operations to the sensor output in order
to present an aesthetically pleasant image, optimized for human vision[36, 39]. Especially in situation in which the
sensor’s exposure is either very short or very long with respect to the illumination, the sensor can behave in a highly
nonlinear fashion.[39]

The nonlinearity is generally not a problem, but any measurements attempting to relate pixel values to the incident
irradiance would quickly break down against expected results based on the system's physics. With a linearized digital
camera a user can use multiple exposures to construct a high dynamic range (HDR) image of a scene, which can provide
quantitative measurements of relative irradiance for each color channel of a camera[39]. Based on the image formation
process in digital cameras[45], with additional corrections for vignetting and fixed-pattern artifacts in the imaging
system, an estimate of the original scene radiance can be computed in a relative scale.

### Measuring linearity
Generally ignoring the effects of noise, saturation and cut-off, when comparing two images of the same scene, whose only
difference is exposure time, the following relationship holds true

$$\frac{g(B_s(x,y))}{t_s} = E(x,y) = \frac{g(B_l(x,y))}{t_l},$$

in which \(g\) is the camera's ICRF and the indices \(s\) and \(l\) denote the image with the shorter and the longer exposure
respectively[]. Essentially this equation just tells us that in the static case the camera sensor measures the same
irradiance in both images and that the exposures of each image are linearly related to each other by their exposure times.

We can approach the quantification of the linearity for a pair of images in two ways. In an absolute scale we can solve
the previous equation for

$$A(x,y) =  g(B_{\text{s}}(x,y)) - \frac{t_{\text{s}}}{t_{\text{l}}} g(B_{\text{l}}(x,y)),$$

where we denote the difference as \(A(x,y)\). We'll call this difference as the absolute disparity image. In the ideal
case without the effects of noise, saturation, cut-off and a perfectly known ICRF, the disparity image would be zero
everywhere. In order to establish a relative scale, we can solve the equation for

$$R(x,y) =  \frac{g(B_{\text{s}}(x,y)) - r g(B_{\text{l}}(x,y))}{r g(B_{\text{l}}(x,y))},$$

in which we define \(r = t_s/t_l\) and denote by \(R(x,y)\) a relative disparity image.

Utilizing the disparity images we propose the following simple metric for measuring the linearity of an imaging system.
We'll calculate the spatial mean and sample standard deviation for \(R(x,y)\), yielding us a value
\(\overline{R} \pm \sigma_{\overline{R}}\). Then for a set of images we'll calculate this value for all unique pairs in the
set to plot the data with \(\overline{R}\) on the y-axis and the exposure time ratio \(t_{\text{s}}/t_{\text{l}}\) on the
x-axis. An absolute and relative scale examples of such plots for the Olympus SC100 camera are in figure \ref{fig:lin}.
Some of the images utilized in calculating the data points are shown in figure \ref{fig:exp_series}. The closer to zero
all the spatial means of the disparity images are, the more accurately the ICRF is known and the less noise there is.


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

