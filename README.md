# Estimating Late Arrival Time for a Hydraulic Conductivity Map

Exploring different techniques to get insights from Hydraulic conductivity map for a relationship between t_95.

- HCM : Hydraulic Conductivity Map

## Introduction

***

![Hydraulic Conductivity Map](raw/map.png)

## Previous Work

***

### Research Papers

1. [Seeing macro-dispersivity from hydraulic conductivity field with convolutional neural network](https://www.sciencedirect.com/science/article/abs/pii/S0309170819308607)
2. [Minimum Hydraulic Resistance and Least Resistance Path in Heterogeneous Porous Media](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2017WR020418)

## Approach

***

## Probabilistic path estimation

***

![Probabilistic Estimation](raw/ProbabilisticEstimation.png)

- We define the tendency of the contaminant to pass from the cell i to one of the 8 neighbor cells j as CONDUCTIVITY, ​ **C_ij**, where rij is the distance between the two cells and Ki and Kj are the hydraulic conductivity of the considered cells. While, ​ **P_ij** ​ is the probability that the contaminant contained in the i-th cell moves in the j-th cell (that is one of the 8 cells around i).

- To generate data for all the maps in dataset it takes >2 minutes.
- To improve the performance we parallelized the code using OMP and OpenMP, based on the below structure to generate probilistic paths.

![Model Architecture](raw/ProbabilisticModel.png)

## Comparing t_95 of different HCM

***

![Deep Neural Network](raw/model.png)

- We create a DNN (deep neural network) with a combination of CNN (convolution neural network) and fully connected layer.
- Below is the model summary,

```bash
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 50, 103, 16)       800       
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 25, 51, 16)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 25, 51, 64)        50240     
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 12, 25, 64)        0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 12, 25, 256)       409856    
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 6, 12, 256)        0         
_________________________________________________________________
flatten (Flatten)            (None, 18432)             0         
_________________________________________________________________
dense (Dense)                (None, 256)               4718848   
_________________________________________________________________
dropout (Dropout)            (None, 256)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 256)               65792     
_________________________________________________________________
dropout_1 (Dropout)          (None, 256)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 512)               131584    
_________________________________________________________________
dense_3 (Dense)              (None, 1)                 513       
=================================================================
Total params: 5,377,633
Trainable params: 5,377,633
Non-trainable params: 0
_________________________________________________________________
```

## Results

We ran the system for different permutations of dropout rate and batch size for a dataset breakdown of 60% 20% 20%. Below are the results,

### Correleration
![Correleration](raw/correlation.png) \

### Actual v/s Predicted for best model
![Actual v/s Predicted for best model](raw/comparisonActualPredictBatchBest.png) \

### Correleration of Best model
![Correleration of Best model](raw/correlationBatch50aD0.5.png) \