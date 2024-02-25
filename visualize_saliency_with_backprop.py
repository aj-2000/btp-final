# -*- coding: utf-8 -*-
"""visualize_saliency_with_backprop_colab.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/github/MisaOgura/flashtorch/blob/master/examples/visualize_saliency_with_backprop_colab.ipynb

## Visualize image-specific class saliency with backpropagation

---

A quick demo of creating saliency maps for CNNs using [FlashTorch 🔦](https://github.com/MisaOgura/flashtorch).


❗This notebook is for those who are using this notebook in **Google Colab**.

If you aren't on Google Colab already, please head to the Colab version of this notebook **[here](https://colab.research.google.com/github/MisaOgura/flashtorch/blob/master/examples/visualise_saliency_with_backprop_colab.ipynb)** to execute.

---

The gradients obtained can be used to visualise an image-specific class saliency map, which can gives some intuition on regions within the input image that contribute the most (and least) to the corresponding output.

More details on saliency maps: [Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps](https://arxiv.org/pdf/1312.6034.pdf).

### 0. Set up

A GPU runtime is available on Colab for free, from the `Runtime` tab on the top menu bar.

It is **recommended to use GPU** as a runtime for the enhanced speed of computation.
"""

# Install flashtorch

!pip install flashtorch

# Download example images

!mkdir -p images

!wget -nv \
    https://github.com/MisaOgura/flashtorch/raw/master/examples/images/great_grey_owl.jpg \
    https://github.com/MisaOgura/flashtorch/raw/master/examples/images/peacock.jpg        \
    https://github.com/MisaOgura/flashtorch/raw/master/examples/images/toucan.jpg         \
    -P /content/images

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'

import matplotlib.pyplot as plt
import torchvision.models as models

from flashtorch.utils import apply_transforms, load_image
from flashtorch.saliency import Backprop

"""### 1. Load an image"""

image = load_image('/content/images/great_grey_owl.jpg')

plt.imshow(image)
plt.title('Original image')
plt.axis('off');

"""### 2. Load a pre-trained Model"""

model = models.alexnet(pretrained=True)

"""### 3. Create an instance of Backprop with the model"""

backprop = Backprop(model)

"""### 4. Visualize saliency maps"""

# Transform the input image to a tensor

owl = apply_transforms(image)

# Set a target class from ImageNet task: 24 in case of great gray owl

target_class = 24

# Ready to roll!

backprop.visualize(owl, target_class, guided=True, use_gpu=True)

"""### 5. What about other birds?

What makes peacock a peacock...?
"""

peacock = apply_transforms(load_image('/content/images/peacock.jpg'))
backprop.visualize(peacock, 84, guided=True, use_gpu=True)

"""Or a toucan?"""

toucan = apply_transforms(load_image('/content/images/toucan.jpg'))
backprop.visualize(toucan, 96, guided=True, use_gpu=True)

"""Please try out other models/images too!"""