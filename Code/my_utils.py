import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from leaspy import Data, Leaspy, IndividualParameters

def split_test_pred_set(df_Test0, ID_Test0): 

    '''
    INPUT: 
        df_Test0 contains all visits for each subejcts in the test set
        ID_Test0 contains the ID of test set
    
    OUTPUT: 
        df_Test contains all visits for each subject excpet the last one (used for personalization)
        df_pred contains the real values of he last visit for each subjects (used for prediction)
    '''
        
    df_Test0 = df_Test0.reset_index()
    df_Test=df_Test0
    df_Pred=df_Test0
    for i_te in ID_Test0:
        df_test_sub=df_Test0[np.in1d(df_Test0['ID'],i_te)]# Dataframe of one test subject
        rows=df_Test[df_Test['ID'] == i_te].index.to_numpy()  # rows of this subject in original test dataframe

        if  (df_test_sub.shape[0])>1: # if this test subject has more than 1 visits the lsat visit will be used for prediction and other visits for test
            #Test data  
            last_row = rows[len(rows) - 1]
            df_Test=df_Test.drop([last_row])#remove last visit from test dataset(which is for prediction dataset)
            #prediction data 
            rows=np.delete(rows,(len(rows)-1))#remove all visits except last visit from prediction dataset
            df_Pred=df_Pred.drop(rows)
        else:
            df_Pred=df_Pred.drop(rows)#if the subject has one visit we remove that row from prediction dataset

    df_Test['ID'] = df_Test['ID'].astype(str)
    df_Test['ID'] = df_Test['ID'].astype(str)


    df_Test = df_Test.set_index(['ID', 'TIME'], verify_integrity=True).sort_index()
    df_Pred = df_Pred.set_index(['ID', 'TIME'], verify_integrity=True).sort_index()

    return df_Test, df_Pred

def param_for_plot(dcm_model, source_dim):

    xi = dcm_model.model.parameters['xi_mean'].numpy()
    tau = dcm_model.model.parameters['tau_mean'].numpy()
    source = dcm_model.model.parameters['sources_mean'].numpy().tolist()
    sources = [np.array(source)*dcm_model.model.source_dimension]
  
    if source_dim==0: 
        parameters = {'xi': xi, 'tau': tau}
    else:   
        parameters = {'xi': xi, 'tau': tau,'sources': sources}
   
        return parameters

def compute_performance_metrics(y_obs, y_pred):
    '''
    INPUT: 
        y_obs contains an array with observed values that we want to predict
        y_pred contains an array with predicted values
    
    OUTPUT: 
        mae the Mean Average Error
        r2 the R squared'''
    mae = np.abs(y_obs - y_pred).mean()
    SSres = ((y_obs-y_pred)**2).sum()
    SStot = ((y_obs-y_obs.mean())**2).sum()

    r2 = 1 - (SSres/SStot)

    return mae, r2

def compute_dcm_model(df, SubList, LabelList, cols, model_type, nb_source, algo_settings, settings_personalization, 
                      timepoints, K=5):
    '''
    INPUT: 
        df = contains the dataframe with all visits for each subjects
        SubList = contains the ID of subjects
        LabelList = contains the labels for each subject
        cols = contains the list of columns to be used as features
        model_type = type of model to be used (e.g., 'logistic')
        nb_source = number of features (e.g., 1)
        algo_settings = settings for the algorithm used for fitting the model
        settings_personalization = settings for the algorithm used for personalizing the model
        timepoints = contains the time points for the trajectory
        K = number of folds for cross-validation
    
    OUTPUT: 
        model_lst = list that contains the fitted models
        ip_df = dataframe that contains the individual parameters dataframe
        AverageTraj = Numpy Array that contains the average trajectory for each fold
        mae_mat = Numpy Array that contains the Mean Average Error for each fold
        r2_mat = Numpy Array that contains the R squared for each fold
    '''
    
    # Initialize variables
    mae_mat = np.zeros((2,K))
    r2_mat = np.zeros((2,K))
    model_lst = list()
    ip_df = pd.DataFrame()
    AverageTraj=np.zeros((len(timepoints), len(cols), K))

    kf = StratifiedKFold(n_splits=K, random_state=0, shuffle=True)

    for fold, (train_idx, test_idx) in enumerate(kf.split(SubList, LabelList)): 
        print(f"Train set: {len(train_idx)}")
        print(f"Test set: {len(test_idx)}")

        # SELECT TRAIN AND TEST DATASET
        ID_Test = [SubList[i] for i in test_idx]
        ID_Train = [SubList[i] for i in train_idx]

        df_Test = df[np.in1d(df.index.get_level_values('ID'), ID_Test)]
        df_Train = df[np.in1d(df.index.get_level_values('ID'), ID_Train)]

        # Split observed and value to predict    
        df_Validation, df_to_Pred = split_test_pred_set(df_Test, ID_Test)

        # Create train and test datasets only with feature columns
        data_train = Data.from_dataframe(df_Train[cols])
        data_validation = Data.from_dataframe(df_Validation[cols])

        ## FIT - ON TRAINING SET
        leaspy_model = Leaspy(model_type, source_dimension=nb_source)
        leaspy_model.fit(data_train, settings=algo_settings)

        average_parameters = param_for_plot(leaspy_model, leaspy_model.model.source_dimension)
        
        ip_average = IndividualParameters()
        ip_average.add_individual_parameters('average', average_parameters)
        values = leaspy_model.estimate({'average': timepoints}, ip_average)
        
        AverageTraj[:,:,fold] = values['average']

        # PERSONALIZATION - ON VALIDATION SET
        individual_paramas = leaspy_model.personalize(data_validation, settings_personalization)
    
        # PREDICTION
        reconstruction = leaspy_model.estimate(df_to_Pred.index, individual_paramas)
        df_pred = reconstruction.add_suffix('_pred') # predicted values

        # SAVE INDIVIDUAL PARAMETERS AS DATAFRAME
        ip_df_new = individual_paramas.to_dataframe()

        # SAVE MAE and R2
        for i, s in enumerate(cols):
            mae_mat[i, fold], r2_mat[i, fold] = compute_performance_metrics(df_to_Pred[s], df_pred[s+'_pred'])
                
        # SAVE MODEL LEASPY
        model_lst.append(leaspy_model)
        ip_df = pd.concat([ip_df, ip_df_new], axis=0)

    return model_lst, ip_df, AverageTraj, mae_mat, r2_mat