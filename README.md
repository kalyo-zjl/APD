# APD
[Attribute-aware Pedestrian Detection in a Crowd](https://arxiv.org/pdf/1910.09188.pdf)

## Installation

To run the demo, the following requirements are needed.
```
numpy
matplotlib
torch >= 0.4.1
glob
argparse
[DCNv2](https://github.com/CharlesShang/DCNv2/tree/pytorch_0.4)
```

## Model
[final.pth](https://drive.google.com/file/d/1CqLsFCLzWaDPojwPlbepeqUhEP7n9nS0/view?usp=sharing) is a model trained on [cityperson dataset](https://bitbucket.org/shanshanzhang/citypersons/src/default/).

## Demo
The demo code and the trained is only for cityperson dataset.
```
python demo.py --img_list 'images/*.pth'
```


## Citation

If you find this project useful for your research, please use the following BibTeX entry.

    @article{DBLP:journals/corr/abs-1910-09188,
      author    = {Jialiang Zhang and
               Lixiang Lin and
               Yun{-}chen Chen and
               Yao Hu and
               Steven C. H. Hoi and
               Jianke Zhu},
      title     = {{CSID:} Center, Scale, Identity and Density-aware Pedestrian Detection
               in a Crowd},
      journal   = {CoRR},
      volume    = {abs/1910.09188},
      year      = {2019},
      url       = {http://arxiv.org/abs/1910.09188},
      archivePrefix = {arXiv},
      eprint    = {1910.09188},
      timestamp = {Tue, 22 Oct 2019 18:17:16 +0200},
      biburl    = {https://dblp.org/rec/bib/journals/corr/abs-1910-09188},
      bibsource = {dblp computer science bibliography, https://dblp.org}
    }
