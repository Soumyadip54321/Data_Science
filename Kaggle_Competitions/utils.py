def Preprocessing(X_tr,X_test,target,n_splits,seed):
    '''
    Function to impute missing values using simple imputer with strategy median across different columns.
    '''
    skf = StratifiedKFold(n_splits=n_splits,shuffle=True,random_state=seed)
    y= X_tr[target]

    # fit on partial data to infer the shape
    sample_transformed = preprocessing.fit_transform(X_tr.iloc[:len(X_tr)//2,:])
    oof_vals = np.zeros((len(X_tr),sample_transformed.shape[1]))


    for tr_idx,val_idx in skf.split(X_tr,y):
        tr_fold = X_tr.iloc[tr_idx]
        va_fold = X_tr.iloc[val_idx]
        # simp_imputer = SimpleImputer(strategy='median')
        preprocessing.fit(tr_fold)
        oof_vals[val_idx] = preprocessing.transform(va_fold)

    preprocessing.fit(X_tr)
    X_test_transformed = preprocessing.transform(X_test)
    X_test = pd.DataFrame(X_test_transformed,columns=preprocessing.get_feature_names_out(),index=X_test.index)

    X_tr = pd.DataFrame(oof_vals,columns=preprocessing.get_feature_names_out(),index=X_tr.index)
    
    return X_tr,X_test