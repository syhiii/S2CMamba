# DCFN
DCFN: A Pan-sharpening Network Incorporating Spatial and Spectral Constraints via Mamba Module
# Dataset
You can download GF2 WV3 QB dataset at [THIS](https://github.com/liangjiandeng/PanCollection)
# dcfn block
✨You can call our proposed modules: CIM, SCM, CMIM directly in **/dcfn_block/block/**.
# sota

✨We keep some sota models in **/sota**, including Gppnn,Msdnn, PanMamba, etc.

**References**

- **(DMLD)** K. Zhang et al., "Learning Deep Multiscale Local Dissimilarity Prior for Pansharpening," in IEEE Transactions on Geoscience and Remote Sensing, vol. 61, pp. 1-15, 2023.[LINK](https://ieeexplore.ieee.org/abstract/document/10210612)
- **(DPFN)** J. Wang, Z. Shao, X. Huang, T. Lu and R. Zhang, "A Dual-Path Fusion Network for Pan-Sharpening," in IEEE Transactions on Geoscience and Remote Sensing, vol. 60, pp. 1-14, 2022, Art no. 5403214, doi: 10.1109/TGRS.2021.3090585.
- **(Gppnn)** S. Xu, J. Zhang, Z. Zhao, K. Sun, J. Liu and C. Zhang, "Deep Gradient Projection Networks for Pan-sharpening," 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), Nashville, TN, USA, 2021, pp. 1366-1375, doi: 10.1109/CVPR46437.2021.
- **(U2Net)** U2Net: A General Framework with Spatial-Spectral-Integrated Double U-Net for Image Fusion, ACM MM 2023.
- **(PanNet)** J. Yang, X. Fu, Y. Hu, Y. Huang, X. Ding and J. Paisley, "PanNet: A Deep Network Architecture for Pan-Sharpening," 2017 IEEE International Conference on Computer Vision (ICCV), Venice, Italy, 2017.
- **(MSDNN)** X. He et al., "Multiscale Dual-Domain Guidance Network for Pan-Sharpening," in IEEE Transactions on Geoscience and Remote Sensing, vol. 61, pp. 1-13, 2023.
- **(PanMamba)** He X, Cao K, Yan K, et al. Pan-Mamba: Effective pan-sharpening with State Space Model, arid preprint arXiv:2402.12192, 2024.
```shell
# run.py can help you run this sota methods
  python run.py 
# -ChDim [WV3:8 GF2:4 QB:4]
# -save_folder [your save path]
# -dataset [WV3 QB GF2]
# -algorithm [pannet panmamba msdnn gppnn dpsn dmld dspnet]
# -mode [If mode=1, it is the training mode, if mode=3 test mode if mode=4 test the effect of the model on full]
# Note that you may need to modify some parameters of the model when mode=4 to fit the full dataset with different sizes of datasets
xxx
```

  
# Train
```shell
# train
xxx
# test
xxx
```


*Waiting for updates*🐛🐛🐛
