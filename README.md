# Keypoint-guided Factorization for Optimal Transport

## Abstract
Optimal transport (OT) theory offers powerful framework for comparing probability distributions by minimizing the cost of transporting mass between them. Despite its board applicability in various fields of machine learning, the practical use of OT is hindered by its computational complexity and statistical performance in noisy, high-dimensional problems. Recent advances resolve these challenges by adding additional structural assumptions on transport with low-rank constraints on either cost/kernel matrices or the feasible set of couplings considered in OT. They serve the low-rank constraints by moving mass through small sets of intermediate points, instead of direct transportation between source and target distribution. However, such modelings impose a type of bottleneck and result in a loss of information that make the estimation of point-wise transport inaccurate. To address this bottleneck issue,  we thus introduce the concept of keypoint guidance to OT factorization. Our approach, **KFOT**, preserves the matching of keypoint pairs over factored couplings and then leverages the relation of each points to the keypoints for guidance towards the point-to-point matching. We show that **KFOT** inherits the robustness of factored OT in noisy, high-dimensional problems while be capable of mitigating the bottleneck issue caused by the factorization framework itself.

![image](https://github.com/minhngt62/ot-kpgf/blob/main/assets/showcase.png)

## Get Started
The dependencies are in [requirements.txt](https://github.com/minhngt62/ot-kpgf/blob/main/requirements.txt). Python 3.8 and above is recommended for the installation of the environment.
```
pip install -r requirements.txt
```

## Experiments
To reproduce the experiments, run the [`notebooks`](https://github.com/minhngt62/ot-kpgf/tree/main/notebooks), collect the results of each experiment in [`logs`](https://husteduvn-my.sharepoint.com/:u:/g/personal/minh_nt204885_sis_hust_edu_vn/EX8tWH_VzR5DpY0KMKn7InMBG7jp8lsDVo0ipnGkVaXjow?e=77Zesh) that corresponds to the experiment name. Then, for the visualization in documentation, please follow the [`0. Visualization.ipynb`](https://github.com/minhngt62/ot-kpgf/blob/main/notebooks/__0.%20Visualization.ipynb).
