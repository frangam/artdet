# ArtDet: Machine Learning Software for Automated Detection of Art Deterioration in Easel Paintings


[![License: GNU 3](https://img.shields.io/badge/License-GNU%203-blue.svg)](https://www.gnu.org/licenses/gpl-3.0) <img src="https://img.shields.io/github/release/frangam/artdet.svg"/> [![GitHub all releases](https://img.shields.io/github/downloads/frangam/artdet/total)](https://github.com/frangam/artdet/releases/download/1.0/artdet-v1.0.zip)

This is the official implementation code of the paper <b>"ArtDet: Machine Learning Software for Automated Detection of Art Deterioration in Easel Paintings"</b> ([`Paper`(https://doi.org/10.1016/j.softx.2024.101917))

[[`Paper`](https://doi.org/10.1016/j.softx.2024.101917)] [[`Dataset`](https://doi.org/10.5281/zenodo.8429815)] [[`BibTeX`](#citation)]




## Installation
### 1. **Clone our repository:**

   ```shell
   git clone https://github.com/frangam/artdet.git
   cd artdet
   ```

### 2. **Set up Python 3.12 and create the environment**

1. **Install Python 3.12** (if not already installed). On macOS or Linux:

   ```shell
   sudo apt update
   sudo apt install -y python3.12 python3.12-venv python3.12-dev
   ```

   For macOS (if using Homebrew):

   ```shell
   brew install python@3.12
   ```

2. **Create a virtual environment** using Python 3.12:

   ```shell
   python3.12 -m venv venv
   ```

3. **Activate the environment**:

   On macOS/Linux:

   ```shell
   source venv/bin/activate
   ```

   On Windows:

   ```shell
   .\venv\Scripts\activate
   ```


### 3. **Install Mask-RCNN updated version to work with Tensorflow 2:**

   ```shell
   git clone https://github.com/alsombra/Mask_RCNN-TF2.git
   cd Mask_RCNN-TF2
   pip install -r requirements.txt
   python setup.py install

   ```

### 4. **Install our custom dependencies:**

   ```shell
   cd ..
   pip install -r requirements.txt
   ```
### 5. **Run the web app:**

   ```shell
   python src/run.py
   ```
   Then, the wep app is running on http://127.0.0.1:5000

## <a name="Dataset"></a>Dataset
You can download our ArtInsight Dataset at:  [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8429815.svg)](https://doi.org/10.5281/zenodo.8429815)

**Place images in this folder:** data/---
data/train/---
data/val/---

## <a name="Models"></a>Model Checkpoints

Click the links below to download the checkpoint for the corresponding model type.

- [v0-tf-format](https://huggingface.co/frangam/artdet-v0/blob/main/model-maskrcnn-tf.zip)
- [v0-h5-format](https://huggingface.co/frangam/artdet-v0/blob/main/model-maskrcnn.h5)

**Locate the downloaded model to this path:** model/---


## <a name="Citation"></a>Citation

If you use our code in your research, please use the following BibTeX entry:

```
@article{GARCIAMORENO2024101917,
title = {ARTDET: Machine learning software for automated detection of art deterioration in easel paintings},
journal = {SoftwareX},
volume = {28},
pages = {101917},
year = {2024},
issn = {2352-7110},
doi = {https://doi.org/10.1016/j.softx.2024.101917},
url = {https://www.sciencedirect.com/science/article/pii/S2352711024002875},
author = {Francisco M. Garcia-Moreno and Jesús Cortés Alcaraz and José Manuel {del Castillo de la Fuente} and Luis Rodrigo Rodríguez-Simón and María Visitación Hurtado-Torres}
}

```

And also cite our Dataset:

(Submitted to)
```

@article{Garcia-MorenoArtInsight,
  title={ArtInsight: A Detailed Dataset for Detecting Deterioration in Easel Paintings},
  author={Garcia-Moreno, Francisco Manuel and del Castillo de la Fuente, Jose Manuel  and Rodríguez-Simón, Luis Rodrigo and Hurtado-Torres, María Visitación},
  year={2024},
journal={Data in Brief}
  doi={pending},
  url={},
}
```

