# Universalization of Any Adversarial Attack using Very Few Test Examples

This repository is the preliminary codebase for the CODS-COMAD 2022 research track paper [Universalization of any adversarial attack using very few test examples](https://dl.acm.org/doi/abs/10.1145/3493700.3493718). Working code with be updated soon.

## Overview
The paper gives a simple SVD based algorithm to obtain an universal attack from well known adversarial directions like Gradients, FGSM and DeepFool directions.


## Dependencies
Codebases used in the paper as is or modified accordingly.

* [DeepFool] (https://github.com/LTS4/DeepFool)


## Code documentation.

* Instructions to construct the SVD-Attack
    * Collect the attack vectors 
    * Obtain the top SVD vectors using the given script
    * Apply the SVD-Attack with scale factor and obtain the fooling rate.
	* Script to generate the SVD vectors 

## Citation

If the code related to our work is useful for your work, kindly cite this work as given below:

```[bibtex]
@inproceedings{kamath2020universalization,
  title={Universalization of Any Adversarial Attack Using Very Few Test Examples}, 
  author={Sandesh Kamath and Amit Deshpande and K V Subrahmanyam and Vineeth N Balasubramanian},
  booktitle = {5th Joint International Conference on Data Science & Management of Data (9th ACM IKDD CODS and 27th COMAD)},
  year = {2022},
  pages = {72–80},
  howpublished={arXiv preprint arXiv:2005.08632},
  url={https://dl.acm.org/doi/abs/10.1145/3493700.3493718}
}

```
