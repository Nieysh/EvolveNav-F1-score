# EvolveNav: Empowering LLM-Based Vision-Language Navigation via Self-Improving Embodied Reasoning


<div align="center">
<a target="_blank" href="https://expectorlin.github.io/">Bingqian Lin</a><sup>1*</sup>,
<a href="https://scholar.google.com/citations?user=jV19-sIAAAAJ" target="_blank">Yunshuang Nie</a><sup>2*</sup>,
<a href="https://openreview.net/profile?id=~Khun_Loun_Zai2" target="_blank">Khun Loun Zai</a><sup>2</sup>,
<a target="_blank" href="http://sadil13.github.io/">Ziming Wei</a><sup>2</sup>,
<a target="_blank" href="https://mingfei.info/">Mingfei Han</a><sup>3</sup>,
<a target="_blank" href="https://rongtao-xu.github.io/">Rongtao Xu</a><sup>3</sup>,
<a href="https://openreview.net/profile?id=~Minzhe_Niu1" target="_blank">Minzhe Niu</a><sup>4</sup>,
<a href="https://scholar.google.com/citations?user=OEPMQEMAAAAJ&hl=en" target="_blank">Jianhua Han</a><sup>4</sup>,
<a target="_blank" href="https://personal.ntu.edu.sg/hanwangzhang/">Hanwang Zhang</a><sup>5</sup>,
<a target="_blank" href="http://www.linliang.net/">Liang Lin</a><sup>2</sup>,
<a target="_blank" href="https://www.sigs.tsinghua.edu.cn/cbk/main.htm">Bokui Chen</a><sup>6&ddagger;</sup>,
<a target="_blank" href="https://www.mvig.org/">Cewu Lu</a><sup>1&ddagger;</sup>,
<a target="_blank" href="https://scholar.google.com/citations?user=voxznZAAAAAJ">Xiaodan Liang</a><sup>2&ddagger;</sup>

<sup>1</sup>Shanghai Jiao Tong University</span>
<sup>2</sup>Sun Yat-Sen University</span>
<sup>3</sup>Mohamed bin Zayed University of Artificial Intelligence</span>
<sup>4</sup>Yinwang Intelligent Technology</span>
<sup>5</sup>Nanyang Technological University</span>
<sup>6</sup>Tsinghua University</span>
<br/>
<sup>*</sup>Equal contribution.
<sup>&ddagger;</sup> Corresponding author.
</br>
</div>

<div align="center">
    <a href="https://arxiv.org/abs/2506.01551" target="_blank">
    <img src="https://img.shields.io/badge/Paper-arXiv-deepgreen" alt="Paper arXiv"></a>
</div>



## :new: Updates
- [06/2025] [Arxiv paper](https://arxiv.org/abs/2506.01551) released.
- [06/2026] Action-reasoning-F1-score computing.


## Installation

The environment installation of EvolveNav follows that in [NaviLLM](https://github.com/zd11024/NaviLLM).

1. Follow instructions [here](https://github.com/peteanderson80/Matterport3DSimulator) to install Matterport3D simulators.

2. Installation requirements for VLN training:
```setup
cd EvolveNav
conda create --name evolvenav python=3.8.16
conda activate evolvenav
pip install -r requirements.txt
```

## Data Preparation
### navigation data, features, and model
   
1. Follow [NaviLLM](https://github.com/zd11024/NaviLLM) to get the navigation data, features.

2. Prepare model checkpoint for evaluation.

## Inference
The model can be tested on NVIDIA A100 or V100 GPUs with ~24G memories. The testing batch size is set as 2 on each GPU. First, inference and record model output on remote server:
```setup
sh scripts/evaluation/eval_r2r_f1score.sh
```
Second, add an api key for vlm judge and compute F1-score locally:
```setup
sh scripts/evaluation/eval_r2r_f1score_local.sh
```

## Acknowledgement
Some of the codes are built upon [NaviLLM](https://github.com/zd11024/NaviLLM), [VLN-DUET](https://github.com/cshizhe/VLN-DUET), and [Tag2Text](https://github.com/xinyu1205/recognize-anything). Thanks for their great works!
