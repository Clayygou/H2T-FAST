## H2T-FAST: Head-to-Tail Feature Augmentation by Style Transfer for Long-Tailed Recognition (ECAI 2023)

A Pytorch implementation of our ECAI 2023 paper "H2T-FAST: Head-to-Tail Feature Augmentation by Style Transfer for Long-Tailed Recognition".

How to train
-----------------

Go into the "example" directory.

   ```
   python main.py --dataset dataset --c config/config of dataset.yaml --strategy strategy

   ```
See 'main.py' for more parameters.


Citation
-----------------

  ```
  @inproceedings{DBLP:conf/ecai/MengGST0X23,
  author       = {Ziyao Meng and
                  Xue Gu and
                  Qiang Shen and
                  Adriano Tavares and
                  Sandro Pinto and
                  Hao Xu},
  editor       = {Kobi Gal and
                  Ann Now{\'{e}} and
                  Grzegorz J. Nalepa and
                  Roy Fairstein and
                  Roxana Radulescu},
  title        = {{H2T-FAST:} Head-to-Tail Feature Augmentation by Style Transfer for
                  Long-Tailed Recognition},
  booktitle    = {{ECAI} 2023 - 26th European Conference on Artificial Intelligence,
                  September 30 - October 4, 2023, Krak{\'{o}}w, Poland - Including
                  12th Conference on Prestigious Applications of Intelligent Systems
                  {(PAIS} 2023)},
  series       = {Frontiers in Artificial Intelligence and Applications},
  volume       = {372},
  pages        = {1712--1719},
  publisher    = {{IOS} Press},
  year         = {2023},
  url          = {https://doi.org/10.3233/FAIA230456},
  doi          = {10.3233/FAIA230456},
  timestamp    = {Fri, 27 Oct 2023 20:40:30 +0200},
  biburl       = {https://dblp.org/rec/conf/ecai/MengGST0X23.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}

  ```
This code is partly based on the open-source implementations from [imbalanced-DL](https://github.com/ntucllab/imbalanced-DL)

