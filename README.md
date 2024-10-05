<a id="readme-top"></a>


<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/alixbird/predict_haq">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a>

<h3 align="center">Predicting function from imaging in Rheumatoid Arthritis</h3>

  <p align="center">
    This project investigates the use of plain radiography of the hands and feet in predicting the HAQ score, a measure of function. 
    We investigate both contemporaneous function at every time point available, as well as prediction of HAQ after 1-2 years based on baseline imaging.
    The perfromance of the AI algorithm is compared to human performance, which is measured by the manual assessment of radiographs
    as per the Sharp van der Heijde scoring method. 
  </p>
</div>

<!-- GETTING STARTED -->
### Installation

1. Clone the repo
  ```
   git clone https://github.com/alixbird/predict_haq.git
  ```
2. Install prerequisties packages
  ```
  pip install -r requirements.txt
  ```
3. Install package 
  ```
  pip install .
  ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- USAGE EXAMPLES -->
## Usage

To train the models and produce ROC curves of the performance, simply run the python script train_haq.py.
Paths to the csv files and images directory need to be specified as well as where you want the checkpoints and figures saved. 
Note this data is not publicly available so this is purely for demonstration. 

```
python bin/train_haq.py --csvpath=/path/to/dataframes/ --imagepath=/path/to/xray_images --checkpointpath=/path/to/checkpoints --figurepath=/path/to/figures
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- RESULTS EXAMPLES -->
## Results examples
TBA

<!-- ACKNOWLEDGMENTS -->
## Supervised by:

* [Professor Lyle Palmer]()
* [Professor Susanna Proudman]()
* [Dr Lauren Oakden-Rayner]()

<p align="right">(<a href="#readme-top">back to top</a>)</p>

