# Summary

The technical reference for this project is split into one section per subpackage. Each package provides distinct 
functionality and the general theme of them is summarized here.

1. The [Common](common.md) subpackage provides general functionalities used across the project. These include functions and
classes for IO operations, general mathematical and statistical operations and enums used with various classes and functions.
2. The [Datasets](datasets.md) subpackage provides dataset classes, which can be used with the PyTorch Dataloader class for managing the
data loading process in functions.
3. The [Inference](inference.md) subpackage provides functionality that can be used to measure a camera's linearity, create HDR images,
linearize single images and compute quantitatively well-defined mean and uncertainty images from a stack of images or
a video.
4. The [Metadata](metadata.md) subpackage provides classes for managing information related to the files, which are managed by the
FrameData and PairedFrameData classes. The BaseMetadata class provides a guideline for implementing a Metadata class,
while the others provide concrete ready-to-use classes to manage image metadata and video metadata.
5. The [Models](models.md) subpackage provides the inverse camera response function model classes, which are built on the torch.nn.module
class. The base model provides a guideline for creating a new ICRF model, while the concrete implementations provide
different approaches to modelling an ICRF.
6. The [Training](training.md) subpackage provides a training loop function for the ICRF models. In addition to the training loop, it also
provides the various functions for loss computation used in the training loop.
7. The [Validation](validation.md) subpackage provides IO and typechecks used internally and thus doesn't expose any functionality
publicly. The subpackage might eventually contain and expose functionality related to model validation.
8. The [Visualization](visualization.md) subpackage provides functions for plotting and visualizing information from various sources
in this project. The package is currently only used internally and exposes no public API, but that might change in the
future.

As the project is still WIP, the organization of these packages might still change. For example, I'm considering merging
the metadata subpackage into the common subpackage, as they are quite closely related. Similarly, if the amount of loss
functions in modules.training.losses increases significantly over time, I might split it into its own subpackage entirely.