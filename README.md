# Implementation of "A Correlation-Based Feature Weighting Filter for Naive Bayes" (Liangxiao Jiang et al.)

## Getting started

Run the following command to install the dependecies:

```
pip install -r requirements.txt
```

The CBFW classificator was implemented following the standards of [Scikit learn](https://scikit-learn.org/). For a example
of usage please refer to the notebook ```example.ipynb```.

## Results

### Accuracy results (%):

| Dataset       | Paper | Mine  |
| ------------- | ----- | ----- |
| anneal        | 98,5  | 91,87 |
| anneal.ORIG   | 94,60 | 91,90 |
| audiology     | 74,22 | 70,15 |
| autos         | 77,95 | 72,81 |
| balance-scale | 73,76 | 90,72 |
| breast-cancer | 72,46 | 72,49 |
| breast-w      | 97,14 | 97,34 |
| colic         | 83,34 | 83,22 |
| colic.ORIG    | 73,70 | 73,05 |
| credit-a      | 86,99 | 86,59 |
| credit-g      | 75,70 | 75,11 |
| diabetes      | 78,01 | 66,82 |
| glass         | 73,37 | 97,79 |
| heart-c       | 82,94 | 82,58 |
| heart-h       | 83,82 | 83,99 |
| heart-statlog | 83,44 | 82,78 |
| hepatitis     | 85,95 | 84,29 |
| hypothyroid   | 98,56 | 98,93 |
| ionosphere    | 91,82 | 91,39 |
| iris          | 94,40 | 94,20 |
| kr-vs-kp      | 93,58 | 93,52 |
| labor         | 92,10 | 89,75 |
| letter        | 75,22 | 75,66 |
| lymphography  | 84,81 | 81,72 |
| mushroom      | 99,19 | 99,88 |
| primary-tumor | 47,20 | 45,05 |
| segment       | 93,47 | 86,90 |
| sick          | 97,36 | 97,26 |
| sonar         | 82,56 | 75,39 |
| soybean       | 93,66 | 92,40 |
| splice        | 96,19 | 96,13 |
| vehicle       | 62,91 | 61,08 |
| votes         | 92,11 | 92,14 |
| vowel         | 68,84 | 62,33 |
| waveform-5000 | 83,11 | 82,10 |
| zoo           | 95,96 | 96,45 |

### Elapsed time (seconds):

| Dataset       | Paper | Mine   |
| ------------- | ----- | ------ |
| audiology     | 5,28  | 2,18   |
| balance-scale | 0,09  | 0,0053 |
| breast-cancer | 0,18  | 0,051  |
| breast-w      | 0,21  | 5,56   |
| colic,ORIG    | 0,99  | 0,43   |
| credit-a      | 0,46  | 0,15   |
| credit-g      | 1     | 0,36   |
| diabetes      | 0,23  | 12,87  |
| glass         | 0,19  | 0,015  |
| heart-c       | 0,22  | 0,031  |
| heart-h       | 0,25  | 0,024  |
| heart-statlog | 0,11  | 0,0093 |
| hepatitis     | 0,25  | 0,029  |
| ionosphere    | 1,69  | 0,15   |
| iris          | 0,12  | 0,0021 |
| kr-vs-kp      | 6,82  | 1,21   |
| letter        | 29,13 | 5,6    |
| lymphography  | 0,35  | 0,032  |
| mushroom      | 12,08 | 7,78   |
| primary-tumor | 0,43  | 0,088  |
| segment       | 4,41  | 0,05   |
| sonar         | 1,59  | 0,069  |
| waveform-5000 | 17,4  | 0,46   |
| zoo           | 0,27  | 0,015  |
