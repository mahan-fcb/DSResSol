from models.SequenceOnlyModel import SequenceOnlyModel
from models.SequenceWithBiologicalModel import SequenceWithBiologicalModel
from lib.Util import Util
import argparse

parser = argparse.ArgumentParser(description='main.py')
parser.add_argument('-s', '--sequence_only', action='store_true',
                    help='train sequence only model, default is biological')


if __name__ == '__main__':
    args = parser.parse_args()
    util = Util()
    if(args.sequence_only):
        model = SequenceOnlyModel(util)
    else:
        model = SequenceWithBiologicalModel(util)
    model.train()
