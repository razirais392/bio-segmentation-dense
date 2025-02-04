# Dense Image Classification for Segmentation


This repository houses codes for segmenting images via per-pixel (or dense) classification.  The core idea is based on semantic segmentation [lon2015] and we adopt the architecture with skip connections proposed by [ron2015].  Currently this code assumes grayscale images and works for binary or multi-class problems.  Note this code was written in the author's spare time (while at UMD) so YMMV.  Also, as with many things in this field, by the time you are reading this there may already be a better native implementation in Keras.

mjp, 2016
(last updated April 2017)

## Quick Start

- I am currently using Python 3 and Keras version 2.0.2 with Theano as the backend.
- To train a U-Net, see [this example](./Examples/ISBI_2012/train_isbi.py)
- To deploy, see [this ipython notebook](./Examples/ISBI_2012/deploy_isbi.ipynb)


### Dimension Ordering

Note that this code makes explicit assumptions about the order of data dimensions (we expect tensors with shape (n_examples, n_channels, n_rows, n_cols)).  If you experience ugly-looking errors from underlying Theano codes you may want to make sure that the Keras dimension ordering is set to "th"; e.g here is my keras.json:

```
{
    "image_dim_ordering": "th",
    "epsilon": 1e-07,
    "floatx": "float32",
    "backend": "theano"
}
```


## References:

1.  [lon2015] Long, Shelhamer, Darrell "Fully Convolutional Networks for Semantic Segmentation." CVPR 2015.
2.  [ron2015] Ronneberger, Fischer, Brox "U-Net: Convolutional Networks for Biomedical Image Segmentation." 2015. https://arxiv.org/abs/1505.04597.
3.  Jocic, Keras implementation of U-Net. https://github.com/jocicmarko/ultrasound-nerve-segmentation.


## Disclaimer
This code is provided "as is" without warranty of any kind, either express or implied, including, but not limited to, the implied warranties of merchantability, fitness for a particular purpose, and non-infringement. In no event will the author be liable for damages of any kind, including without limitation any special, indirect, incidental, or consequential damages even if the author has been advised of the possibility of such damages.
