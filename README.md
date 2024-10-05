<a id="readme-top"></a>

[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]


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
    The perfromance of the AI algorithm is compared to human performance, which is considered to be the manual assessment of radiographs
    as per the Sharp van der Heijde scoring method. 
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

[![Product Name Screen Shot][product-screenshot]](https://example.com)

Here's a blank template to get started: To avoid retyping too much info. Do a search and replace with your text editor for the following: `github_username`, `repo_name`, `twitter_handle`, `linkedin_username`, `email_client`, `email`, `project_title`, `project_description`

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->
## Getting Started

This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.


### Installation

1. Clone the repo
   ```
   git clone https://github.com/github_username/repo_name.git
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
Paths to the csv files, images need to be specified as well as where you want the checkpoints and figures saved. 
Note this data is not publicly available so this is purely to demonstrate ease of use.

```
python bin/train_haq.py --csvpath=/hpcfs/users/a1628977/data/dataframes/ --imagepath=/hpcfs/users/a1628977/data/xray_images --checkpointpath=/hpcfs/users/a1628977/predict_haq/checkpoints --figurepath=figures
 ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- RESULTS EXAMPLES -->
## Results examples


<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* [Professor Lyle Palmer]()
* [Professor Susanna Proudman]()
* [Dr Lauren Oakden-Rayner]()

<p align="right">(<a href="#readme-top">back to top</a>)</p>

