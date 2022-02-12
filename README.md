*NOTE:* This file is a template that you can use to create the README for your project. The *TODO* comments below will highlight the information you should be sure to include.

## Mobile Phone Price Range

In this project, I will try to find the best model and parameters for Mobile price classification problem using Azure AutoML and HyperDrive modules. Then, I will deploy the best model. With this project, using various mobile data, one will be able to estimate the price range of desired mobile price by providing desired features. I perform two different approaches to find the best model for this price range classification problem. 

## Dataset

### Overview
I am using Mobile price classification dataset from Kaggle [Kaggle Link](https://www.kaggle.com/iabhishekofficial/mobile-price-classification) as an external dataset. 

### Task
This dataset contains 20 features of a mobile phone including battery power, dual sim, clock speed, 4G, memory, etc. With given features, target, which is price range, will bep redicted as a classification problem, where value of 0(low cost), 1(medium cost), 2(high cost) and 3(very high cost). So, by entering desired 20 features, one can predict the possible price range of the desired mobile.

### Access
By uploding this data to dataset section in Azure ML portal, we can access it easily through workspace in our notebooks. By using ```key = "Mobile Phone Data"``` or any dataset name entered while uploading instead of "Mobile Phone Data". Then access is simply using the over workspace.datasets ```dataset = ws.datasets[key]```

## Automated ML
Here, first, early stopping parameters were set as 20 minutes for experiment timeout, max concurrent iterations set as 5, and primary metric was selected as `accuracy`
```
automl_settings = {
    "experiment_timeout_minutes": 20,
    "max_concurrent_iterations": 5,
    "primary_metric" : 'accuracy'
}
```

Then, task is provided as classification, where label specified as 'price_range' and early stopping as True (to satisfy above conditions). Featurization selected as auto to perform auto feature engineering.

``` automl_config = AutoMLConfig(compute_target=compute_target,
                             task = "classification",
                             training_data=dataset,
                             label_column_name="price_range",   
                             enable_early_stopping= True,
                             featurization= 'auto',
                             debug_log = "automl_errors.log",
                             **automl_settings
                            )
```
### Results

Best model was found as XGBoostClassifier with parameters as 
* booster='gbtree'
* colsample_bytree=0.6
* eta=0.5
* gamma=0
* max_depth=6
* max_leaves=63
* n_estimators=100
* n_jobs=1
* objective='reg:logistic'

with an `accuracy` score of `0.9505`. This is pretty high and much better than the HyperDrive I run for the other experiment. This score can be improved maybe very slightly if I cange the early stopping parameters like making `experiment_timeout_minutes` from 20 to 60 or 120 or so to allow AutoML to perform more model training with more parameters. 

Here is the result of `RunDetails` widget:
![AutoML RunDetails](/Images/Image1.png)

And best model trained with its parameters:
![AutoML bestmodel](/Images/Image2.png)

## Hyperparameter Tuning

I chose `RandomForestClassifier` for the experiment since it is a powerful classifier that uses decision tress approach for varying multiple splits. It is usually one of the most used models for classification tasks. It is fast and efficient. For hyperparameter tuning over this model, I used `n_estimators`, `criterion`, and `max depth` as parameters to be explored. These determine how many estimators will be used, what will be the evaluation criteria at nodes and maximum depth of splitting, respectively. I used `RandomParameterSampling` as a parameter sampling method, where `n_estimators` will be picked randomly between 0 and 1000, `criterion` as randomly picked between 'gini' and 'entropy', and finally, `max_depth` to be selected randomly between 0 and 10. 

As an early termination policy, `BanditPolicy` is used with `slack_factor=0.1` only.

### Results

Best metrics, `accuracy`, score was obtained as `0.885`, which is pretty smaller than the score we obtained with Auto ML. Parameters were found as `963` for `n_estimators`, `gini` for `criterion`, and `7` for `max_depth`. This can be improved by using better model like Gradient Boosting methods like XGBoost or by adding more parameters and more max total runs (here I set it as `max_total_run=4` to make it fast) to allow system to work on more parameters.


Here is the result of `RunDetails` widget:
![Hyper RunDetails](/Images/Image3.png)

And best model trained with its parameters:
![Hyper bestmodel](/Images/Image4.png)

## Model Deployment

XGBoostClassifier from AutoML was found to have the highest metrics score. It was deployed using `score.py` that receive the data at endpoint and predicts the `price_range` using the model deployed. Then, writes the prediction output. Azure Container Instace with `cpu_cores=2` and `memory_gb=2` was used for the endpoint model prediction computation.

Here you can see the model is deployed successfully and it is active,

on the notebook:

![Deploy_not](/Images/Image5.png)

on the portal with endpoint
![Deploy_port](/Images/Image6.png)


In order to receive model predictions, once can prepare values for 20 features in json format to the endpoint by using request library and its post function as `request.post(endpoint_url, data)`(URL is the REST endpoint that is shown in image above) or simply can use the `service.run(data)` where the service is the variable that built on `Mode.deploy(...)` for the deployment.

Here is the dmeonstration of model prediction from the endpoint
![endpoint](/Images/Image7.png)

## Screen Recording
Link for the screencast is:
[Screencast Link](https://drive.google.com/file/d/1AAN5ZHg8FthHB9wemlWKFrHnQIKJlbBW/view?usp=sharing)

