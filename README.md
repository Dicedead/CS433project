# Modelling Energetic Particles in Matter
![Compton effect](https://github.com/Dicedead/CS433project/raw/main/project2/plots/emission_drawing.png)

## Reproducibility
Get the data from the [drive](https://drive.google.com/drive/u/1/folders/1Zsz5ZGmZoPcBf4f30cwVOi3FtWQaU5GS?sort=13&direction=a). The easiest is to download the pickled ```water_dataset.pkl``` and to place it in ```project2/pickled_data```. Then, run ```test.py``` located in ```project2/src```. For quick results, we recommend setting the number of particles ```NMC``` to 100,000 or less.

## Folder organization

### project2/src
Contains all the machine learning related aspects of the code. The file ```cgan.py``` contains superclasses for cGAN generators, critics and hyperparameters, as well as a few other convenience methods. Then, each component of the ML system has a file inside the same folder. To re-train a model, one can run its corresponding file.

### project2/src_data
Contains files for data parsing and handling. 

### project2/model_parameters
Saves models to disk for future loading.
