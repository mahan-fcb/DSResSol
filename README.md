- Steps
  1. Download Anaconda of your OS in: https://www.anaconda.com/download
  2. Make sure you are in DsResSol directory
  3. RUN `conda env create -f environment.yml`
  4. RUN `conda activate DsResSol`
  5. RUN `python main.py --sequence_only` to train the model with sequence only
  6. RUN `python main.py` to train the model with sequence and biological features
