# 📄 Visual Implicit Geometry Transformer for Autonomous Driving

Official implementation of the paper: **"Visual Implicit Geometry Transformer for Autonomous Driving"**  

---

## 🔗 Links

- 📄 **Paper:** [![arXiv](https://img.shields.io/badge/arXiv-2602.05573-b31b1b?logo=arxiv)](https://www.arxiv.org/abs/2602.05573)
- 🤗 **Hugging Face Model:** Coming soon!
- 🌐 **Project Page (optional):** Coming soon!  

---

## Get started

### 1) Create environment

```bash
conda create -y -n vigt python=3.10
conda activate vigt
conda install -y -c nvidia cuda-nvcc=12.4 cuda-cudart-dev=12.4 cuda-cccl=12.4 cuda-libraries-dev=12.4
```

### 2) Install Python dependencies

```bash
python -m pip install -r requirements.txt
python -m pip install --no-build-isolation --no-binary=nerfacc nerfacc
```

### 3) Download nuScenes mini and unpack it to any local folder, for example `./data/`.

### 4) Download model checkpoint
1. Download the [ViGT checkpoint file](PASTE_CHECKPOINT_LINK_HERE).

Checkpoint summary:

| Checkpoint | Train datasets | Chamfer Dist (nuScenes) | Chamfer Dist (nuPlan) | Chamfer Dist (Waymo) | Chamfer Dist (AV2) | mIoU (Occ3D nuScenes) |
|---|---|---:|---:|---:|---:|---:|
| [ViGT checkpoint file](PASTE_CHECKPOINT_LINK_HERE) | nuScenes, nuPlan, Waymo, Av2 | 1.8727 | 3.286 | 2.3846 | 3.008 | 0.5591 |

### 5) Run Jupyter demo
Open `examples/nuscenes_demo.ipynb`.
---

## Citation

If you found our work useful, please cite it with the following bibtex.

```
@article{vigt2026,
  title   = {Visual Implicit Geometry Transformer},
  author  = {Arsenii Shirokov, Mikhail Kuznetsov, Danila Stepochkin, Egor Evdokimov, Daniil Glazkov, Nikolay Patakin, Anton Konushin, Dmitry Senushkin},
  journal = {arXiv preprint arXiv:2602.05573},
  year    = {2026}
}
```
