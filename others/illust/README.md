# illust

### Dependencies
python 3.6

In addition, please add the project folder to PYTHONPATH and `conca install` the following packages:
- `Pytorch`

**AttnGAN**

Pytorch implementation for reproducing AttnGAN results in the paper [AttnGAN: Fine-Grained Text to Image Generation
with Attentional Generative Adversarial Networks](https://arxiv.org/pdf/1711.10485.pdf) by Tao Xu, Pengchuan Zhang, Qiuyuan Huang, Han Zhang, Zhe Gan, Xiaolei Huang, Xiaodong He. (This work was performed when Tao was an intern with Microsoft Research). 

<img src="https://github.com/taoxugit/AttnGAN/blob/master/framework.png" width="900px" height="350px"/>


**Data**

- Collect urls of image file:

  - For irastoya dataset: ```python3.6 irasutoya-search.py --keyword 人 --start 0 --end 2300 2> irasutoya-human.log 1> irasutoya-human.txt```

  - For irastoya dataset: ```python3.6 irasutoya-year-caption.py --year 2018 2> log-2018.txt 1> tee list-2018.txt```
