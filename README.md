DSResSol : A sequence-based solubility predictor created with Dilated Squeeze Excitation Residual Networks

Abstract

Motivation: Protein solubility is an important thermodynamic parameter critical for the characterization of a protein’s function, and a key determinant for the production yield of a protein in both the research setting and within industrial (e.g. pharmaceutical) applications. Thus, a highly accurate in silico bioinformatics tool for predicting protein solubility from protein sequence is sought. In this study, we developed a deep learning sequence-based solubility predictor, DSResSol, that takes advantage of the integration of squeeze excitation residual networks with dilated convolutional neural networks. The model captures the frequently occurring amino acid k-mers and their local and global interactions, and highlights the importance of identifying long-range interaction information between amino acid k-mers to achieve higher performance in comparison to existing deep learning-based models. 
Result: DSResSol uses protein sequence as input, outperforming all available sequence-based solubility predictors by at least 5% in accuracy when the performance is evaluated by two different independent test sets. Compared to existing predictors, DSResSol not only reduces prediction bias for insoluble proteins but also predicts soluble proteins within the test sets with an accuracy that is at least 13% higher. We derive the key amino acids, dipeptides, and tripeptides contributing to protein solubility, identifying glutamic acid and serine as critical amino acids for protein solubility prediction. Overall, DSResSol can be used for fast, reliable, and inexpensive prediction of a protein’s solubility to guide experimental design.
Availability: The source code, datasets, and web server for this model is available at https://tgs.uconn.edu/dsres_sol


- Steps
  1. Download Anaconda of your OS in: https://www.anaconda.com/download
  2. Make sure you are in DsResSol directory
  3. RUN `conda env create -f environment.yml`
  4. RUN `conda activate DsResSol`
  5. RUN `python main.py --sequence_only` to train the model with sequence only
  6. RUN `python main.py` to train the model with sequence and biological features

If you have any question please contact me. mohammad73madani73@gmail.com
