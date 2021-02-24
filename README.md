# GravitationalLenses

The code and data provided in this project creates a machine learning algorithm, a convolutional neural network (CNN), to identify strong gravitational 
lenses within [DES](https://www.darkenergysurvey.org). The data contains images that either contain gravitational lenses or not.
Since gravitational lensing systems are so rare, artificial lenses were created to train our CNN, as to what these images look like. 

We create artificial images of sources being lensed using code from [Thomas Collett](https://github.com/tcollett) in his 
[LensPop](https://github.com/tcollett/LensPop) repository. This is referred to as `positive images`. 

We clip the 10,000 * 10,000 pixel images from DES to size of 100 * 100 pixels, and these images are referred to as  the `negative images`, 
most of which do not contain a gravitational lens, as lenses are so rare. 

Random 100 * 100 snippets of the original DES images are added to the `positive images` to create more realistic lensing images, by adding random sky from 
these DES images to them. 

The `positive images` and `negative images` are use in training the CNN to learn the difference between the images containing gravitational lenses and those that do not.

To create negative images:
1. Run [negativeDES.py](https://github.com/Annarien/GravitationalLenses/blob/main/Training/negativeDES.py) and its utils file [negativeDESUtils.py](https://github.com/Annarien/GravitationalLenses/blob/main/Training/negativeDESUtils.py)

To create positive images;
1. Run [positiveSet.py](https://github.com/Annarien/GravitationalLenses/blob/main/Training/positiveSet.py) and its utils file [positiveSetUtils.py](https://github.com/Annarien/GravitationalLenses/blob/main/Training/positiveSetUtils.py)

To run the Keras CNN:
1. Run [KerasCNN.py](https://github.com/Annarien/GravitationalLenses/blob/main/Training/KerasCnn.py)

A few of these resources were kindly made available by [Thomas Collett](https://github.com/tcollett) in his [LensPop](https://github.com/tcollett/LensPop) repository.
