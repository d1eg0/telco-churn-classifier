Telco customer churn
====================
The goal of this script is to predict Telco customer churn. It trains a GBM classifier using the dataset of Telco customer crunch.
Implements in Python (with scikit-learn) the ML Azure approach https://gallery.cortanaintelligence.com/Experiment/Telco-Customer-Churn-5 

The raw dataset is imbalanced. Hence, the SMOTE algorithm is used to balance the datasets. SMOTE is included in the library http://contrib.scikit-learn.org/imbalanced-learn
The supervised classification model used is the Gradient Boosting Classifier (https://en.wikipedia.org/wiki/Gradient_boosting).

The next figure shows the performance gain applying SMOTE in the preprocessing stage:

![alt tag](Telco_GB_SMOTE.png)
