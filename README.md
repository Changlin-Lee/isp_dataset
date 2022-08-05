# A Well-aligned Dataset for Learning Image Signal Processing on Smartphones from a High-end Camera (Siggraph Poster 2022)
This is the data preprocessing code for paper A Well-aligned Dataset for Learning Image Signal Processing on
Smartphones from a High-end Camera

[Yazhou Xing], [Changlin Li], [Xuaner Zhang], [Qifeng Chen]

In Siggraph Poster 2022.
**[Demo Video]()** | **[Project Page]()** | **[Paper]()**


## Requirement
Our alignment use pwcnet-pytorch(https://github.com/NVlabs/PWC-Net/tree/master/PyTorch) for flow refinement.
Please follow the pwcnet-pytorch implementation at first.

```
$ pip install -r requirements.txt
```

## Align the patches by the following procedure
```
$ python alignment_refine_sift.py
$ python alignment_refine_flow.py
$ python alignment_realign_sift.py
```
