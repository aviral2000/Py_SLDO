# Py_SLDO
A Python testing suite for ideas relating to learning stable sparse differential operators from data.

ArXiv link with the pre-print associated with this work: https://arxiv.org/abs/2405.00198.

Key points for using this framework:
1. The current state of the code is problem-specific
2. The Examples folder contains all the numerical experiments that are included in the arXiv preprint.
3. Each script handles all aspects of the framework: data generation, pre-analysis of stability plots, determining learned sparse differential operators (both LDOs and S-LDOs) for different stencil sizes and generating the figures for the article
4. The Python scripts are written to be executed in block style as often done using IDEs like Spyder.
5. The Python script can be executed in one go, but relevant sections of the code that display figures and animation should be commented out.
6. The framework is robust to data generation parameters and initial conditions. The default parameters are those used to generate the figures in the article.

Ongoing extensions:
1. Demonstrations of reduced order modeling using learned differential operators

Please do not hestitate to contact the code contributors if there are any questions regarding the theory, implementation or potential collaborations of using this method for specific applications.
