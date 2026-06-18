# HELIME
After installing lime, in lime/lime_tabular.py replace  
def explain_instance(self,  
                         data_row,  
                         predict_fn,  
                         labels=(1,),  
                         top_labels=None,  
                         num_features=10,  
                         num_samples=5000,  
                         distance_metric='euclidean',  
                         model_regressor=None,  
                         sampling_method='gaussian')    
  with   
  def explain_instance_helime(self,   
                                data_row,                              
                                x_star,  
                                categorical_names,    
                                categorical_features,  
                                feature_names,                 
                                predict_fn,  
                                labels=(1,),  
                                top_labels=None,  
                                num_features=10,  
                                num_samples=5000,  
                                distance_metric='euclidean',  
                                model_regressor=None,                                             
                                n_radial=10,  
                                n_fi=10)    
                         
                         

## Prelimanary requirements

In order to run this project, it is recommended to have installed `Python 3.10+`  
All the modules required to run the project are specified in the `requirements.txt` file.  
The list of the main libraries used is as follows:
- `numpy` – numerical operations and vector/matrix manipulation;
- `pandas` – loading and processing datasets;
- `scikit-learn` – training machine learning models and evaluating performance;
- `lime` – generating local explanations for model predictions;
- `matplotlib` – generating plots and visualizing results;
- `seaborn` – statistical visualizations;
- `scipy` – scientific calculations and mathematical functions used by some libraries;
- `joblib` – saving and loading trained models;
- `tensorflow` – build, train evaluate machine learning and deep learning models;
- `keras` – framework for neural network models;
- `imblearn` – handling imbalanced datasets using techniques such as oversampling, undersampling, or combined methods.  
The exact versions of the modules are listed in the `requirements.txt` file to ensure reproducible results.

### Install
```bash
pip install -r requirements.txt
```
To check the installed modules, you can run:
```bash
pip list
```
### Requirements.txt file
```bash
pip install -r requirements.txt
```
---
## Project Availability and Future Collaborations
The complete project, including the source code, experimental scripts, documentation, dependency files, and result-related materials, can be made available upon request.  
For access to the complete project or for inquiries regarding future research collaborations, please contact:  
Gabriela Moise: gmoise@upg-ploiesti.ro

