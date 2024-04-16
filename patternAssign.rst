.. code:: ipython3

    import pandas as pd
    data = pd.read_csv('adult.csv.csv')
    print (data)


.. parsed-literal::

           |1x3 Cross validator     Unnamed: 1  Unnamed: 2     Unnamed: 3  \
    0                        25        Private      226802           11th   
    1                        38        Private       89814        HS-grad   
    2                        28      Local-gov      336951     Assoc-acdm   
    3                        44        Private      160323   Some-college   
    4                        18              ?      103497   Some-college   
    ...                     ...            ...         ...            ...   
    16276                    39        Private      215419      Bachelors   
    16277                    64              ?      321403        HS-grad   
    16278                    38        Private      374983      Bachelors   
    16279                    44        Private       83891      Bachelors   
    16280                    35   Self-emp-inc      182148      Bachelors   
    
           Unnamed: 4           Unnamed: 5          Unnamed: 6       Unnamed: 7  \
    0               7        Never-married   Machine-op-inspct        Own-child   
    1               9   Married-civ-spouse     Farming-fishing          Husband   
    2              12   Married-civ-spouse     Protective-serv          Husband   
    3              10   Married-civ-spouse   Machine-op-inspct          Husband   
    4              10        Never-married                   ?        Own-child   
    ...           ...                  ...                 ...              ...   
    16276          13             Divorced      Prof-specialty    Not-in-family   
    16277           9              Widowed                   ?   Other-relative   
    16278          13   Married-civ-spouse      Prof-specialty          Husband   
    16279          13             Divorced        Adm-clerical        Own-child   
    16280          13   Married-civ-spouse     Exec-managerial          Husband   
    
                    Unnamed: 8 Unnamed: 9  Unnamed: 10  Unnamed: 11  Unnamed: 12  \
    0                    Black       Male            0            0           40   
    1                    White       Male            0            0           50   
    2                    White       Male            0            0           40   
    3                    Black       Male         7688            0           40   
    4                    White     Female            0            0           30   
    ...                    ...        ...          ...          ...          ...   
    16276                White     Female            0            0           36   
    16277                Black       Male            0            0           40   
    16278                White       Male            0            0           50   
    16279   Asian-Pac-Islander       Male         5455            0           40   
    16280                White       Male            0            0           60   
    
              Unnamed: 13   target  
    0       United-States   <=50K.  
    1       United-States   <=50K.  
    2       United-States    >50K.  
    3       United-States    >50K.  
    4       United-States   <=50K.  
    ...               ...      ...  
    16276   United-States   <=50K.  
    16277   United-States   <=50K.  
    16278   United-States   <=50K.  
    16279   United-States   <=50K.  
    16280   United-States    >50K.  
    
    [16281 rows x 15 columns]
    

.. code:: ipython3

    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.naive_bayes import GaussianNB
    from sklearn.metrics import accuracy_score
    
    data = pd.read_csv('adult.csv.csv')
    print(data.head())
    data['target'] = (data['target'] == ' >50K.').astype(int)
    X = data.drop('target', axis=1)
    y = data['target']
    X = pd.get_dummies(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    naive_bayes = GaussianNB()
    naive_bayes.fit(X_train, y_train)
    y_pred = naive_bayes.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    


.. parsed-literal::

       |1x3 Cross validator  Unnamed: 1  Unnamed: 2     Unnamed: 3  Unnamed: 4  \
    0                    25     Private      226802           11th           7   
    1                    38     Private       89814        HS-grad           9   
    2                    28   Local-gov      336951     Assoc-acdm          12   
    3                    44     Private      160323   Some-college          10   
    4                    18           ?      103497   Some-college          10   
    
                Unnamed: 5          Unnamed: 6  Unnamed: 7 Unnamed: 8 Unnamed: 9  \
    0        Never-married   Machine-op-inspct   Own-child      Black       Male   
    1   Married-civ-spouse     Farming-fishing     Husband      White       Male   
    2   Married-civ-spouse     Protective-serv     Husband      White       Male   
    3   Married-civ-spouse   Machine-op-inspct     Husband      Black       Male   
    4        Never-married                   ?   Own-child      White     Female   
    
       Unnamed: 10  Unnamed: 11  Unnamed: 12     Unnamed: 13   target  
    0            0            0           40   United-States   <=50K.  
    1            0            0           50   United-States   <=50K.  
    2            0            0           40   United-States    >50K.  
    3         7688            0           40   United-States    >50K.  
    4            0            0           30   United-States   <=50K.  
    Accuracy: 0.7930610991710163
    

.. code:: ipython3

    TP = sum((y_pred == 1) & (y_test == 1))
    TN = sum((y_pred == 0) & (y_test == 0))
    FP = sum((y_pred == 1) & (y_test == 0))
    FN = sum((y_pred == 0) & (y_test == 1))
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    
    print("Sensitivity (True Positive Rate):", sensitivity)
    print("Specificity (True Negative Rate):", specificity)
    


.. parsed-literal::

    Sensitivity (True Positive Rate): 0.31057563587684067
    Specificity (True Negative Rate): 0.9366533864541833
    

.. code:: ipython3

    probs = naive_bayes.predict_proba(X_test)
    likelihood_over_50K = probs[:, 1]
    prior_over_50K = sum(y_train) / len(y_train)
    evidence = sum(likelihood_over_50K * prior_over_50K)
    posterior_over_50K = likelihood_over_50K * prior_over_50K / evidence
    print("Posterior probability of making over 50K a year:", posterior_over_50K)
    


.. parsed-literal::

    Posterior probability of making over 50K a year: [1.07755904e-05 2.32283441e-05 3.24133039e-05 ... 4.43184472e-06
     2.16579600e-05 2.98033960e-06]
    

