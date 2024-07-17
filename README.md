# DS4EO_Project_24SS
The classification project accoding to the Sentinel-2 image and EU-forests dataset(take the subset of Germany).

# Pipeline
1. For Machine Learning, Preprocessing -> Feature Extraction -> Classification -> Evaluation
2. Deep Learning: Preprocessing -> Model -> results
3. Essential Depencies: Sci-Kit Learn, PyTorch, torchvision, numpy, matplotlib, umap-learn
4. Notice: run all scripts in the project directory, all relative files' path is based on the project directory.

# "Data"
## Given files
1. File1.csv: EPSG:3035, grid coords and its species' name
2. File2.csv: Count of the Species Names
3. Test Set(Geojson) in Evaluation

## First_10 species
First 10 species from Germany dataset, "_norepeat" means there is no grid with multiple labels;
## All_10 species
First 10 species from all EU forests dataset(EU_forest.csv, EU_withcountry.csv with the column of countries)

## ksh
Dataset from other people.

## file_pre.py, read_file.py
Deal with the csv files, mainly to filter.

# "GEE"
The Javascript programs run on the Google Earth Engine.

# "Inital Model" 
Beginning test for ML model, involves:

1. loader: class for read jeojson files, convert the geojson to numpy, and deal with null or other situations;
2. Augm: Implement the Data Augmentation(first tries)
3. Data: Some processed Numpy files.
4. Feature_selection: results(supports of Functions) of Feature selection in numpy.
5. old_data: use unfiltered data to train
6. SFFS_test: Do the sffs, use pca to convert each band to three features.

# "Encoder"
Mainly build and train the Autoencoder to extract features;

1. Autoencoder: class of encoder, decoder and autoencoder;
2. train encoder: train the autoencoder
3. classifier & clf_ci: Connect the encoder with classifier model
4. cnn: connect the encoder with cnn

# "Trian4EU"
Mainly consists of CNN part

1. outnull: try to remove all imgs with null in bands;
2. pca: process data with PCA;
3. cnn: build and train CNN;
4. runs & fugure: results and figures of progress
5. loader: the data load class, and change the numpy to torch.tensor

# "test_model"
The implementation of final model;

1. Classifier: main program, involve the classifier model;
2. runs & data: saved numpy/tensor/cnn/figures
3. Eu4cnn: CNN model test
4. augm, augm_new:  Data augmentation
5. pca, fs, NLDA: Feature Extraction programs

# "with_NDVI"
add other parameters to original model: NDVI, CI:

# "Old_test"
First evaluation for our model, just to make sure the test data can be as input of our model

# "PCA_test"
Do some non-linearly dimensionality reduction, and draw figures of results
