import logging
from abc import ABC, abstractmethod
import os.path
import pandas as pd

class AbstractDataLoader(ABC):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def load_data(self, filename):
        logging.info('Checking file exists.')

        if not os.path.isfile(filename):
            logging.error('File does not exist')
            # TODO: raise exception
            raise Exception('file {} does not exist'.format(filename))
        else:
            logging.info('Found file: ' + filename)

class FileDataLoader(AbstractDataLoader):

    # Initialization
    def __init__(self, filename: str):
        super().__init__()
        logging.info('Initializing Data Loading')
        self.filename = filename
        
    # Load data from file and return data
    def load_data(self):
        # TODO: Check file exist
         
        if  os.path.isfile(self.filename):

            # TODO: Load data from file
            logging.info('Loading data using pandas')
            data = pd.read_csv(self.filename, index_col = 0)
            # TODO: Return your data object here
            return data
        
        else:
            logging.error('File does not exist')
            raise Exception('file {} does not exist'.format(self.filename))

#if __name__ == '__main__':
#    test = FileDataLoader('C:/Users/Ying-Fang.Kao/Documents/causal_inference/interview-test-final-stats-causalinference/data/dataset.csv')
#    data = test.load_data()