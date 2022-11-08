**Title**: **H**ow does **U**nsupervised **P**re-training **A**ffect **F**ine-tuning on **T**ime **S**eries **C**lassification? (HUPAFTSC)

### Overview

In this code repository, we disclose the model structure of the time series encoder, the code for each pre-training task, and the hyperparameters for pre-training. **If the paper is accepted, we will publish the complete code that can be used to reproduce all the results we show in the paper.**

We also provide:

- detailed datasets information of the 'Using extra pre-training data' analysis section.
- comprehensive results associated with Table 2 for each dataset.
- figures depicting how fast each pre-training task converges on both the MTS and UTS datasets.

### The structure of time-series encoders

It is worth noting that the in_features of the input layer of the model are related to the dimensionality of the dataset, so they will vary depending on the dataset, and the following model structure is just an example of a certain UTS dataset.

#### Lstm (Parameter count: 565600,  the parameter count of under-fitted Lstm is 704)

Code for Lstm can be referred to 'core/ts_nn/rnn.py'

```bazaar
LSTM(
  (lstm): LSTM(1, 128, num_layers=2, batch_first=True, bidirectional=True)
)
```
#### Dialted-Conv (Parameter count: 594868,  the parameter count of under-fitted Dialted-Convis 3320)

Code for Dialted-Conv can be referred to 'core/ts_nn/dilated_conv.py'

```bazaar
DilatedConvEncoder(
  (input_fc): Linear(in_features=1, out_features=64, bias=True)
  (net): Sequential(
    (0): ConvBlock(
      (conv1): SamePadConv(
        (conv): Conv1d(64, 64, kernel_size=(3,), stride=(1,), padding=(1,))
      )
      (conv2): SamePadConv(
        (conv): Conv1d(64, 64, kernel_size=(3,), stride=(1,), padding=(1,))
      )
    )
    (1): ConvBlock(
      (conv1): SamePadConv(
        (conv): Conv1d(64, 64, kernel_size=(3,), stride=(1,), padding=(2,), dilation=(2,))
      )
      (conv2): SamePadConv(
        (conv): Conv1d(64, 64, kernel_size=(3,), stride=(1,), padding=(2,), dilation=(2,))
      )
    )
    (2): ConvBlock(
      (conv1): SamePadConv(
        (conv): Conv1d(64, 64, kernel_size=(3,), stride=(1,), padding=(4,), dilation=(4,))
      )
      (conv2): SamePadConv(
        (conv): Conv1d(64, 64, kernel_size=(3,), stride=(1,), padding=(4,), dilation=(4,))
      )
    )
    (3): ConvBlock(
      (conv1): SamePadConv(
        (conv): Conv1d(64, 64, kernel_size=(3,), stride=(1,), padding=(8,), dilation=(8,))
      )
      (conv2): SamePadConv(
        (conv): Conv1d(64, 64, kernel_size=(3,), stride=(1,), padding=(8,), dilation=(8,))
      )
    )
    (4): ConvBlock(
      (conv1): SamePadConv(
        (conv): Conv1d(64, 64, kernel_size=(3,), stride=(1,), padding=(16,), dilation=(16,))
      )
      (conv2): SamePadConv(
        (conv): Conv1d(64, 64, kernel_size=(3,), stride=(1,), padding=(16,), dilation=(16,))
      )
    )
    (5): ConvBlock(
      (conv1): SamePadConv(
        (conv): Conv1d(64, 64, kernel_size=(3,), stride=(1,), padding=(32,), dilation=(32,))
      )
      (conv2): SamePadConv(
        (conv): Conv1d(64, 64, kernel_size=(3,), stride=(1,), padding=(32,), dilation=(32,))
      )
    )
    (6): ConvBlock(
      (conv1): SamePadConv(
        (conv): Conv1d(64, 64, kernel_size=(3,), stride=(1,), padding=(64,), dilation=(64,))
      )
      (conv2): SamePadConv(
        (conv): Conv1d(64, 64, kernel_size=(3,), stride=(1,), padding=(64,), dilation=(64,))
      )
    )
    (7): ConvBlock(
      (conv1): SamePadConv(
        (conv): Conv1d(64, 64, kernel_size=(3,), stride=(1,), padding=(128,), dilation=(128,))
      )
      (conv2): SamePadConv(
        (conv): Conv1d(64, 64, kernel_size=(3,), stride=(1,), padding=(128,), dilation=(128,))
      )
    )
    (8): ConvBlock(
      (conv1): SamePadConv(
        (conv): Conv1d(64, 64, kernel_size=(3,), stride=(1,), padding=(256,), dilation=(256,))
      )
      (conv2): SamePadConv(
        (conv): Conv1d(64, 64, kernel_size=(3,), stride=(1,), padding=(256,), dilation=(256,))
      )
    )
    (9): ConvBlock(
      (conv1): SamePadConv(
        (conv): Conv1d(64, 64, kernel_size=(3,), stride=(1,), padding=(512,), dilation=(512,))
      )
      (conv2): SamePadConv(
        (conv): Conv1d(64, 64, kernel_size=(3,), stride=(1,), padding=(512,), dilation=(512,))
      )
    )
    (10): ConvBlock(
      (conv1): SamePadConv(
        (conv): Conv1d(64, 300, kernel_size=(3,), stride=(1,), padding=(1024,), dilation=(1024,))
      )
      (conv2): SamePadConv(
        (conv): Conv1d(300, 300, kernel_size=(3,), stride=(1,), padding=(1024,), dilation=(1024,))
      )
      (projector): Conv1d(64, 300, kernel_size=(1,), stride=(1,))
    )
  )
  (repr_dropout): Dropout(p=0.1, inplace=False)
)
```
#### Tstransformer (Parameter count: 397696)

Code for Dialted-Conv can be referred to 'core/ts_nn/transformer_encoder.py'

```bazaar
TSTransformerEncoder(
  (project_inp): Linear(in_features=1, out_features=128, bias=True)
  (pos_enc): FixedPositionalEncoding(
    (dropout): Dropout(p=0.1, inplace=False)
  )
  (transformer_encoder): TransformerEncoder(
    (layers): ModuleList(
      (0): TransformerBatchNormEncoderLayer(
        (self_attn): MultiheadAttention(
          (out_proj): NonDynamicallyQuantizableLinear(in_features=128, out_features=128, bias=True)
        )
        (linear1): Linear(in_features=128, out_features=256, bias=True)
        (dropout): Dropout(p=0.1, inplace=False)
        (linear2): Linear(in_features=256, out_features=128, bias=True)
        (norm1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (norm2): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (dropout1): Dropout(p=0.1, inplace=False)
        (dropout2): Dropout(p=0.1, inplace=False)
      )
      (1): TransformerBatchNormEncoderLayer(
        (self_attn): MultiheadAttention(
          (out_proj): NonDynamicallyQuantizableLinear(in_features=128, out_features=128, bias=True)
        )
        (linear1): Linear(in_features=128, out_features=256, bias=True)
        (dropout): Dropout(p=0.1, inplace=False)
        (linear2): Linear(in_features=256, out_features=128, bias=True)
        (norm1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (norm2): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (dropout1): Dropout(p=0.1, inplace=False)
        (dropout2): Dropout(p=0.1, inplace=False)
      )
      (2): TransformerBatchNormEncoderLayer(
        (self_attn): MultiheadAttention(
          (out_proj): NonDynamicallyQuantizableLinear(in_features=128, out_features=128, bias=True)
        )
        (linear1): Linear(in_features=128, out_features=256, bias=True)
        (dropout): Dropout(p=0.1, inplace=False)
        (linear2): Linear(in_features=256, out_features=128, bias=True)
        (norm1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (norm2): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (dropout1): Dropout(p=0.1, inplace=False)
        (dropout2): Dropout(p=0.1, inplace=False)
      )
    )
  )
  (dropout1): Dropout(p=0.1, inplace=False)
)
```

### The structure of classification head

Code for the classification head can be referred to 'core/ts_nn/classifier.py'

```bazaar
(classifier_head): Sequential(
    (0): Linear(in_features=300, out_features=300, bias=True)
    (1): BatchNorm1d(300, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
    (3): Dropout(p=0.02, inplace=False)
    (4): Linear(in_features=300, out_features=300, bias=True)
    (5): BatchNorm1d(300, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (6): ReLU()
    (7): Dropout(p=0.02, inplace=False)
    (8): Linear(in_features=300, out_features=3, bias=True)
  )
```

### Pre-training hyperparameters
#### Ts2Vec

Code for the pre-training task Ts2Vec can be referred to 'core/ts_pretrain/methods/ts2vec.py'

| Hyperparameter           | Description                                                  | Selected Value | Grid Searched Values                |
| ------------------------ | ------------------------------------------------------------ | -------------- | ----------------------------------- |
| alpha                    | A coefficient that balances Instance Contrastive Loss and Temporal Contrastive Loss. | 0.5            | 0.0, 0.3, 0.5, 0.7                  |
| Augmentation mask scheme | How do we choose to mask the Augmented input.                | binomial       | 'binomial','continuous','mask_last' |
| Output dimension         | Dimension of pre-training head output at each time step $t$. | 32             | -                                   |

#### Ts-Tcc

Code for the pre-training task Ts-Tcc can be referred to 'core/ts_pretrain/methods/ts_tcc.py'

| Hyperparameter                      | Description                                                  | Selected Value | Grid Searched Values |
| ----------------------------------- | ------------------------------------------------------------ | -------------- | -------------------- |
| $\lambda_1$                         | The weight of temporal loss                                  | 0.5            | 0.5, 1.0             |
| $\lambda_2$                         | The weight of contrastive loss                               | 1.0            | 0.5, 1.0             |
| Ratio of temporal $k$               | Controls from which time step (proportional to the overall length) the prediction is made for temporal loss. | 0.4            | 0.2, 0.4             |
| Ratio of max temporal sample number | Controls how many time steps are predicted in total  for temporal loss. | 32             | 32, 64               |
| Contrastive $t$                     | Temperature for  contrastive loss                            | 0.5            | -                    |

#### Mvts

Code for the pre-training task Mvts can be referred to 'core/ts_pretrain/methods/mvts.py'

| Hyperparameter      | Description                                                  | Selected Value | Grid Searched Values |
| ------------------- | ------------------------------------------------------------ | -------------- | -------------------- |
| Masking ratio       | Proportion of time series sequence to be masked              | 0.45           | 0.15, 0.3, 0.45      |
| Mean masking length | Average length of masking subsequences (streaks of 0s)       | 5              | 3, 5, 7              |
| Mask scheme         | Whether each variable should be masked separately ('separate'), or all variables at a certain positions should be masked concurrently ('concurrent') | separate       | -                    |
| Mask distribution   | Whether each masking sequence element is sampled independently at random, or whether  sampling follows a Markov Chain (and thus is stateful), resulting in geometric distributions of  masked squences of a desired mean length. | geometric      | -                    |

#### Srlt

Code for the pre-training task Srlt can be referred to 'core/ts_pretrain/methods/srlt.py'

| Hyperparameter            | Description                                                  | Selected Value | Grid Searched Values |
| ------------------------- | ------------------------------------------------------------ | -------------- | -------------------- |
| Number of negative sample | Number of negative samples per batch example.                | 3              | 3, 4, 5              |
| Negative penalty          | Multiplicative coefficient for the negative sample        loss. | 2              | 1, 1.5, 2            |
| Min length                | Minimum length of a subseries                                | 8              | 2, 4, 8              |

### Datasets for validating the effectiveness of pre-training with additional data

| Dataset type | Contained Datasets and the sizes of them (include both training and test set) |
| ------------ | ------------------------------------------------------------ |
| Food         | Beef (60),Coffee (56),Strawberry (983),Wine (111),OliveOil (60),Ham (214) |
| Image        | Herring (128),MixedShapesRegularTrain (2925) ,MixedShapesSmallTrain (2525),ArrowHead (211),BeetleFly (40),BirdChicken (40),FaceAll (2250),FaceFour (112) |
| ECG          | ECG200 (200), ECG5000 (5000),ECGFiveDays (884),TwoLeadECG (1162),CinCECGTorso (1420),NonInvasiveFetalECGThorax1 (3765),NonInvasiveFetalECGThorax2 (3765) |

### Detailed results of all pre-training methods on the test set

- [Accuracy (early-stopping)](Results_on_Each_Dataset_accuracy_early_stopping.md)
- [Accuracy (last epoch)](Results_on_Each_Dataset_accuracy_last_epoch.md)
- [Min training loss](Results_on_Each_Dataset_min_training_loss.md)

### Plots of convergence speed of all pre-training methods (including training loss and test accuracy)

- [Convergence Speed on UTS and MTS](Convergence_Speed_on_UTS_and_MTS.md)
