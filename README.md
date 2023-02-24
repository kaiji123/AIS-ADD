# Using Segmentation in Counter Strike: Conditional zero

This project tries to use Image Segmentation techniques in a FPS game counter strike.
It uses Semantic Segmentation technique in which a custom model has been created to segment images acquired from the game.
However, when our project aims to use Semantic Segmentation to detect only 1 enemy. Since when there are multiple enemies, we also want to segment the instance. To do this we used Panoptic Segmentation. We have a repository deveoped by facebook team called Maskformer which exactly uses this technique. We will start developing our own code from there. The core idea is to use both techniques and find a bridge to connect. In our case, we believe that the inference time in Semantic Segmentation is better than Panoptic Segmentation and want to use only when there is one enemy. To connect these two ideas, we use a CNN model to differentiate those above mentioned instances. We believe this technique performs better than only using Panoptic Segmentation in terms of time of inference. Several other data such as training accuracy, test accuracy will be also provided.

Below is the general bibliography and link to Maskformer

# Bibliography.

## License

Shield: [![CC BY-NC 4.0][cc-by-nc-shield]][cc-by-nc]

The majority of MaskFormer is licensed under a
[Creative Commons Attribution-NonCommercial 4.0 International License](LICENSE).

[![CC BY-NC 4.0][cc-by-nc-image]][cc-by-nc]

[cc-by-nc]: http://creativecommons.org/licenses/by-nc/4.0/
[cc-by-nc-image]: https://licensebuttons.net/l/by-nc/4.0/88x31.png
[cc-by-nc-shield]: https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg


However portions of the project are available under separate license terms: Swin-Transformer-Semantic-Segmentation is licensed under the [MIT license](https://github.com/SwinTransformer/Swin-Transformer-Semantic-Segmentation/blob/main/LICENSE).

## <a name="CitingMaskFormer"></a>Citing MaskFormer

If you use MaskFormer in your research or wish to refer to the baseline results published in the [Model Zoo](MODEL_ZOO.md), please use the following BibTeX entry.

```BibTeX
@inproceedings{cheng2021maskformer,
  title={Per-Pixel Classification is Not All You Need for Semantic Segmentation},
  author={Bowen Cheng and Alexander G. Schwing and Alexander Kirillov},
  journal={NeurIPS},
  year={2021}
}
```



## MaskFormer: Per-Pixel Classification is Not All You Need for Semantic Segmentation

[Bowen Cheng](https://bowenc0221.github.io/), [Alexander G. Schwing](https://alexander-schwing.de/), [Alexander Kirillov](https://alexander-kirillov.github.io/)

[[`arXiv`](http://arxiv.org/abs/2107.06278)] [[`Project`](https://bowenc0221.github.io/maskformer)] [[`BibTeX`](#CitingMaskFormer)]

<div align="center">
  <img src="https://bowenc0221.github.io/images/maskformer.png" width="100%" height="100%"/>
</div><br/>

# our code
# how to run the model
