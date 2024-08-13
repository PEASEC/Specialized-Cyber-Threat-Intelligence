# Specialized-Cyber-Threat-Intelligence

This repository hosts the code to the model training technique including a novel pipeline combining transfer learning, data augmentation, and few-shot learning for developing an effective specialized cyber threat intelligence (CTI) classifier and Novel techniques of data augmentation and few-shot learning to deal with a small number of training instances.

Further explanations contains the paper [*Multi-Level Fine-Tuning, Data Augmentation, and Few-Shot Learning for Specialized Cyber Threat Intelligence*](https://arxiv.org/abs/2207.11076) [1].

[1]: Bayer, Frey and Reuter (2022) Multi-Level Fine-Tuning, Data Augmentation, and Few-Shot Learning for Specialized Cyber Threat Intelligence,


## Contact and Support

If you have any questions, need access to datasets or the complete research data, or if you encounter any bugs, please feel free to contact me!



# Citing

If you chose to use any of the techniques or the code itself, please cite the following paper.

```
@misc{https://doi.org/10.48550/arxiv.2207.11076,
  doi = {10.48550/ARXIV.2207.11076},
  url = {https://arxiv.org/abs/2207.11076},
  author = {Bayer, Markus and Frey, Tobias and Reuter, Christian},
  keywords = {Cryptography and Security (cs.CR), Computation and Language (cs.CL), FOS: Computer and information sciences, FOS: Computer and information  sciences},
  title = {Multi-Level Fine-Tuning, Data Augmentation, and Few-Shot Learning for Specialized Cyber Threat Intelligence},
  publisher = {arXiv},
  year = {2022},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```
# Installation

This repository was coded in python version 3.7.11.\
Before using the code make sure to install packages in the right order:
1. cuda/11.1
2. cuDNN/8.3.1
3. requirements.txt
4. `git submodule update --init --recursive` to load the dataset
5. In the file src/data/read_dataset.py add your twitter api informations

The versions of the packages are the ones we used during our evaluations.
The cybersecurity domain-trained CySecBERT model used is published in [CySecBERT: A Domain-Adapted Language Model for the Cybersecurity Domain](https://arxiv.org/abs/2212.02974) Bayer et al. (2022) [2].

[2]: Bayer, M., Kuehn, P.D., Shanehsaz, R., & Reuter, C.A. (2022). CySecBERT: A Domain-Adapted Language Model for the Cybersecurity Domain. ArXiv, abs/2212.02974.

Part of the repository and the techniques is ADAPET explained in the paper [Improving and Simplifying Pattern Exploiting Training](https://arxiv.org/abs/2103.11955) from Tam et al. (2021)[3].

[3]: Tam, D., Menon, R. R., Bansal, M., Srivastava, S., & Raffel, C. (2021). Improving and simplifying pattern exploiting training. arXiv preprint arXiv:2103.11955


# Contributions

Markus Bayer \
Tobias Frey \
Christian Reuter 

# License
BSD 2-Clause License

Copyright (c) 2022 Markus Bayer, Science and Technology for Peace and Security
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

[![License](https://img.shields.io/badge/License-BSD_2--Clause-orange.svg)](https://opensource.org/licenses/BSD-2-Clause)
