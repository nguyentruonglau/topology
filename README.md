[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]


<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github.com/nguyentruonglau">
    <img src="https://github.com/nguyentruonglau/topology/blob/main/images/logo.png" alt="Logo" width="410" height="189">
  </a>

  <h3 align="center">TOPOLOGY</h3>

  <p align="center">
    Topology and Geometric Transformations Applications.
    <br />
    <a href="https://github.com/nguyentruonglau/topology/blob/main/README.md"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/nguyentruonglau/topology/blob/main/README.md">View Demo</a>
    ·
    <a href="https://github.com/nguyentruonglau/topology/issues">Report Bug</a>
    ·
    <a href="https://github.com/nguyentruonglau/topology/pulls">Request Feature</a>
  </p>
</p>



<!-- TABLE OF CONTENTS -->
<details open="open">
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
    <li><a href="#contributing">Contributor</a></li>
    <li><a href="#license">License</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

Topology and Geometric Transformations applications.


### Built With

* [Tensorflow](https://www.tensorflow.org)
* [Keras](https://keras.io)
* [OpenCV](https://opencv.org)
* [Scikit-Learn](https://scikit-learn.org)


### Requirements

* h5py==2.10.0
* imutils==0.5.4
* Keras==2.4.3
* matplotlib==3.3.4
* numpy==1.19.5
* opencv-contrib-python==4.5.1.48
* opencv-python==4.5.2.52
* scikit-image==0.17.2
* scikit-learn==0.24.2
* scipy==1.5.4
* tqdm==4.60.0
* tensorflow==2.5.

<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running follow these simple example steps.


### Prerequisites

1. Setting GPU: for Window at [here](https://www.tensorflow.org/install/gpu#windows_setup) & for Linux at [here](https://www.tensorflow.org/install/gpu#ubuntu_1804_cuda_110) [for Tensorflow].

2. Checking GPU:
   ```
   >> import tensorflow as tf
   
   >> tf.config.list_physical_devices('GPU')
   
   return
   
   >> [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
   
   is successful.
   ```

### Installation

1. Clone the repo
   ```
   >> https://github.com/nguyentruonglau/topology
   ```
2. Install packages
   ```
   >> python -m venv <virtual environments name>
   
   >> activate.bat (in scripts folder)
   
   >> pip install -r requirements.txt
   ```

<!-- USAGE EXAMPLES -->
## Usage

Adjust the parameters at [here](https://github.com/nguyentruonglau/topology/blob/main/main.py) to change:
* Number of images used
* Model architecture
* Dataset

Run code:
```
>> python main.py
```

<!-- CONTRIBUTING -->
## Contributor

1. BS. Nguyen Truong Lau (launt@fsoft.com.vn)
2. PhD. Thai Trung Hieu (hieutt42@fsoft.com.vn)
3. PhD. Pham Tri Cong (congpt3@fsoft.com.vn)
4. PhD. Nicholas George Bishop (@soton.ac.uk)
5. PhD. Thomas Davies (@soton.ac.uk)
6. Assoc. Prof. Tran Thanh Long (Long.Tran-Thanh@warwick.ac.uk)


<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[license-shield]: https://github.com/nguyentruonglau/topology/blob/main/images/license.svg
[license-url]: https://github.com/nguyentruonglau/topology/blob/main/LICENSE.txt
[linkedin-shield]: https://github.com/nguyentruonglau/topology/blob/main/images/linkedin.svg
[linkedin-url]: https://www.linkedin.com/in/lautruongnguyen/
