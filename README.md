# Predicting-Terrorist-Attack-Using-Machine-Learning

[![Codacy Badge](https://api.codacy.com/project/badge/Grade/0788aefc820741f8a08b4adea4235b96)](https://app.codacy.com/app/kunyuhe/Predicting-Terrorist-Attack-Using-Machine-Learning?utm_source=github.com&utm_medium=referral&utm_content=ZIYU-DEEP/Predicting-Terrorist-Attack-Using-Machine-Learning&utm_campaign=Badge_Grade_Dashboard) [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ZIYU-DEEP/Predicting-Terrorist-Attack-Using-Machine-Learning/master?filepath=%2Fnotebooks%2FEvaluate%20Best%20Models.ipynb)

## To Reproduce Our Results

Change you working directory to the folder where you want the project. Clone the repository to your local with:

```console
$ git clone git@github.com:ZIYU-DEEP/Predicting-Terrorist-Attack-Using-Machine-Learning.git
```



**NOTE**: the cloning process might be slower than expected as the repo contains cross-validation predicted probabilities to give users a warm start.



Then, run one of the following:

- Windows

```console
$ chmod u+x run.sh
$ run.sh
```

- Linux

```console
$ chmod +x script.sh
$ ./run.sh
```



**NOTE**: users can alter input parameters in the shell script to change the behavior of the scripts for Feature Engineering (`featureEngineering.py`) and Training (`train.py`). You can use:

```console
$ python featureEngineering.py -h
$ python train.py -h
```

to check what's user inputs are available for each script. **Remember that `--start_clean=0` is mandatory for a warm start.** However, if you changed the random seed or the hyperparameter grid, please change it to `--start_clean=1` to obtain the best classifiers under the modified context. 



To retrieve the best models, the recommended model and their test performances, please run [this notebook](https://hub.gke.mybinder.org/user/ziyu-deep-predi-achine-learning-elswzdv1/notebooks/notebooks/Evaluate%20Best%20Models.ipynb).



### Final Project Submission Check List

- [x] [Final Project Report](<https://github.com/ZIYU-DEEP/Predicting-Terrorist-Attack-Using-Machine-Learning/blob/master/Predicting_Future_Terrorist_Attacks_with_State_of_Art_Machine_Learning_Techniques.pdf>);
- [x] [Code Files](<https://github.com/ZIYU-DEEP/Predicting-Terrorist-Attack-Using-Machine-Learning/tree/master/codes>) and [Notebooks](<https://github.com/ZIYU-DEEP/Predicting-Terrorist-Attack-Using-Machine-Learning/tree/master/notebooks>);
- [x] [Train-Test Pair Description](https://github.com/ZIYU-DEEP/Predicting-Terrorist-Attack-Using-Machine-Learning/blob/master/data/train%20test%20sets/train_test_sets_desc.PNG);
- [x] [List of Features](https://github.com/ZIYU-DEEP/Predicting-Terrorist-Attack-Using-Machine-Learning/blob/master/processed_data/supervised_learning/List%20of%20Features.csv);
- [x] [Feature Importances of the Recommended Model](https://github.com/ZIYU-DEEP/Predicting-Terrorist-Attack-Using-Machine-Learning/tree/master/evaluations/best%20models/viz/Feature%20Importances);
- [x] [Final Lists of Predicted "positive" Entities](https://github.com/ZIYU-DEEP/Predicting-Terrorist-Attack-Using-Machine-Learning/tree/master/evaluations/best%20models/predictions/invervention_lists)
