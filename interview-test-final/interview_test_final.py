import logging
import json
from collections import namedtuple
import os
import pandas as pd
import matplotlib.pyplot as plt
from util.conf import file_path
from util.statistical_tests import one_sample_t_test, two_sample_t_test, two_sample_variance_test
from util.DataLoaders import FileDataLoader
from util.data_processing import DataFrameImputer, add_dummies
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import classification_report    

def sort_file_paths(project_name: str):
    # figure out the path of the file we're runnning
    runpath = os.path.realpath(__file__)
    print(runpath)
    # trim off the bits we know about (i.e. from the root dir of this project)
    rundir = runpath[:runpath.find(project_name) + len(project_name) + 1]
    print(rundir)
    # change directory so we can use relative filepaths
    os.chdir(rundir)
    # The original line could be a bug
    #os.chdir(rundir + project_name)

def load_config():
    run_configuration_file = '../resources/interview-test-final.json'
    with open(run_configuration_file) as json_file:
        json_string = json_file.read()
        run_configuration = json.loads(json_string,
                                       object_hook=lambda d: namedtuple('X', d.keys())(*d.values()))
    return run_configuration

def data_cleansing(data):
    '''Clean the data according to some discoveries found during data exploration'''
    new_data = data.copy()
    
    # it does not really drop the duplicates of 1048161, because they have different spend
    new_data = new_data.drop_duplicates()
    
    # drop all the data from this corrupted user_id
    new_data = new_data.drop(index = 1048161)
    
    # drop duplicate spend (many users have exactly same spend in 17/18)
    # it is unreasonable to recover those corrupted spend data
    new_data = new_data.drop_duplicates(subset = ['spent_17'])
    
    # drop some ages (ages beyond this range are likely to be data error)
    new_data = new_data.loc[(new_data['age']>=18) & (new_data['age']<= 80), :]
    
    # impute categorical with the most frequent, and others with mean
    # TODO: if time allows, try more sophistiated imputing method (ex sampling from the distribution)
    new_data = DataFrameImputer().fit_transform(new_data)
    return new_data
#%%
    
if __name__ == '__main__':
    # Initialize logging
    logging.basicConfig(format="%(asctime)s;%(levelname)s;%(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
    logging.info('Starting classification program')
 
    # Actions: get into working directory, load project config, create dated directories
    sort_file_paths(project_name='interview-test-final')
    run_configuration = load_config()


    # TODO: Load the data by instantiating the FileDataLoader, handle file doesn't exist.
    data_loader = FileDataLoader(file_path)  # Candidate , instantiate your class here
    data = data_loader.load_data()
    
    #%% Data Cleasing and Preprocessing
    new_data = data_cleansing(data)
    
    category_columns = []
    for column in new_data:
        if new_data.dtypes[column] == 'object':
            category_columns.append(column)
    
    dummy_list = category_columns + ['test']
    # drop tea and coffee for having too many missing value
    # TODO: better imputation for tea and coffee consumption
    # TODO: turining bear and exercise mins to binary variables
    exclude_columns =['mins_beerdrinking_year', 'mins_exercising_year', 'tea_per_year','coffee_per_year', 'spent_18', 'great_customer_class']

    features = new_data.drop(columns = exclude_columns)
    features = add_dummies(features, dummy_list)
    target = new_data['great_customer_class']
    
    #%% Random Forest
    # randomly generate a test set (10%)
    features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.1, stratify = target)
    
    # use cross validation to compare a few hyperparameters
    number_of_trees = [100, 500, 1000]
    criterions = ['entropy','gini']
    max_depths = [10, 30, 50]
    min_samples_splits = [5,15,30]
    min_samples_leaves = [1,5,10]
    bootstraps = [True, False]
    
    hyperparameter_score = pd.DataFrame([], columns = 
                                        ['number_of_trees','criterions','max_depths','min_samples_splits','min_samples_leaves', 'bootstraps', 'score'])
    i = 0
    for number_of_tree in number_of_trees:
        for criterion in criterions:
            for max_depth in max_depths:
                for min_samples_split in min_samples_splits:
                    for min_samples_leaf in min_samples_leaves:
                        for bootstrap in bootstraps:
                            RF = RandomForestClassifier(number_of_tree, criterion = criterion, max_depth = max_depth, 
                                                        min_samples_split = min_samples_split, min_samples_leaf = min_samples_leaf, bootstrap = bootstrap)
                            scores = cross_validate(RF, features_train, target_train,  
                                                    scoring = 'f1', cv=5, n_jobs=4, return_train_score = False)
                            hyperparameter_score.loc[i,:] = [number_of_tree, criterion, max_depth, min_samples_split, min_samples_leaf, bootstrap, scores]
                            i = i + 1
    # Best parameters
    hyperparameter_score['test_score']= hyperparameter_score['score'].apply(lambda x: x['test_score'].mean())
    hyperparameter_score = hyperparameter_score.sort_values(by = 'test_score', ascending = False)      

    ## Use the whole training set
    parameters = hyperparameter_score.iloc[0]
    
    RF = RandomForestClassifier(parameters['number_of_trees'], criterion = parameters['criterions'], 
                                max_depth = parameters['max_depths'], min_samples_split = parameters['min_samples_splits'],
                               min_samples_leaf = parameters['min_samples_leaves'], bootstrap = parameters['bootstraps'])
    RF.fit(features, target)   

    # feature importances
    plt.figure(figsize = (15,17))
    feature_importance = pd.Series(RF.feature_importances_, index = features.columns)
    feature_importance = feature_importance.sort_values(ascending = False)
    plt.barh(feature_importance.index, feature_importance)
               
    #%%    
    # TODO: Do the rest of your work here, or in other classes that are called here.
    #model = None  # Candidate, instantiate your class here

    logging.info('Completed program')
