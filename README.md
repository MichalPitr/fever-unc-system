# Combine-FEVER-NSMN
This repository provides an implementation for the paper [Combining Fact Extraction and Verification with Neural Semantic Matching Networks](https://arxiv.org/abs/1811.07039) (AAAI 2019 and EMNLP-FEVER Shared Task Rank-1 System).

# Installation
To build a Docker image of this repository you'll need to have Docker installed. You may need root priviliges. Building the repository may take a while and requires around 50 GB of disk space. Here's more info about docker build command https://docs.docker.com/engine/reference/commandline/run/.
```
docker build --tag NAME https://github.com/MichalPitr/fever-unc-system.git
```
--tag gives image a name.
# Running it
There's multiple ways to run a docker image, but here's the most basic one. To learn more about Docker's CLI visit https://docs.docker.com/engine/reference/commandline/run/.
```
docker run NAME
```


## Citation
If you find this implementation helpful, please consider citing:
```
@inproceedings{nie2019combining,
  title={Combining Fact Extraction and Verification with Neural Semantic Matching Networks},
  author={Yixin Nie and Haonan Chen and Mohit Bansal},
  booktitle={Association for the Advancement of Artificial Intelligence ({AAAI})},
  year={2019}
}
```
