# ArtDet: Machine Learning Software for Automated Detection of Art Deterioration in Easel Paintings


[![License: GNU 3](https://img.shields.io/badge/License-GNU%203-blue.svg)](https://www.gnu.org/licenses/gpl-3.0) <img src="https://img.shields.io/github/release/frangam/artdet.svg"/> [![GitHub all releases](https://img.shields.io/github/downloads/frangam/artdet/total)](https://github.com/frangam/artdet/releases/download/1.0/artdet-v1.0.zip)

This is the official implementation code of the paper <b>"ArtDet: Machine Learning Software for Automated Detection of Art Deterioration in Easel Paintings"</b> ([`Paper`](Soon))

[[`Paper`]Soon] [[`Dataset`](https://doi.org/10.5281/zenodo.8429815)] [[`BibTeX`](#citation)]




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


## <a name="Dataset"></a>Dataset
You can download our ArtInsight Dataset at:  [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8429815.svg)](https://doi.org/10.5281/zenodo.8429815)

## <a name="Models"></a>Model Checkpoints

Click the links below to download the checkpoint for the corresponding model type.

- [last](soon)





## <a name="Citation"></a>Citation

If you use our code in your research, please use the following BibTeX entry:

```
@article{Garcia-Moreno-PWPF,
  title={ArtDet: Machine Learning Software for Automated Detection of Art Deterioration in Easel Paintings},
  author={Garcia-Moreno, Francisco Manuel and Cortés Alcázar, Jesús and del Castillo de la Fuente, Jose Manuel  and Rodríguez-Simón, Luis Rodrigo and Hurtado-Torres, María Visitación},
  year={2024},
  journal={pending},
  doi={pending},
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

