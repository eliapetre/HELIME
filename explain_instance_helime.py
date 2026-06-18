def explain_instance_helime(self, data_row,                                
                                x_star,
                                categorical_names,  
                                categorical_features,
                                feature_names,               
                                predict_fn,
                                labels=(1,),
                                top_labels=None,
                                num_features=10,
                                num_samples=5000,
                                distance_metric='euclidean',
                                model_regressor=None,                                          
                                n_radial=10,
                                n_fi=10):
                                
        
        """Generates explanations for a prediction.

        First, we generate neighborhood data 
          (see __data_inverse). We then learn locally weighted
        linear models on this neighborhood data to explain each of the classes
        in an interpretable way (see lime_base.py).

        Args:
            data_row: 1d numpy array or scipy.sparse matrix, corresponding to a row
            predict_fn: prediction function. For classifiers, this should be a
                function that takes a numpy array and outputs prediction
                probabilities. For regressors, this takes a numpy array and
                returns the predictions. For ScikitClassifiers, this is
                `classifier.predict_proba()`. For ScikitRegressors, this
                is `regressor.predict()`. The prediction function needs to work
                on multiple feature vectors (the vectors randomly perturbed
                from the data_row).
            labels: iterable with labels to be explained.
            top_labels: if not None, ignore labels and produce explanations for
                the K labels with highest prediction probabilities, where K is
                this parameter.
            num_features: maximum number of features present in explanation
            num_samples: size of the neighborhood to learn the linear model
            distance_metric: the distance metric to use for weights.
            model_regressor: sklearn regressor to use in explanation. Defaults
                to Ridge regression in LimeBase. Must have model_regressor.coef_
                and 'sample_weight' as a parameter to model_regressor.fit()
            Args specific for HELIME method:
            n_radial: is used to compute step to calculate a partition in [0,1]
            n_fi: is used to compute step to calculate a partition in [0,\pi] and [0,2\pi]

            
        Returns:
            An Explanation object (see explanation.py) with the corresponding
            explanations.
        """
        if sp.sparse.issparse(data_row) and not sp.sparse.isspmatrix_csr(data_row):
            # Preventative code: if sparse, convert to csr format if not in csr format already
            data_row = data_row.tocsr()
        ###############################################################################
        
        ############################################################################################
        ############################################################################################
        ##HE-LIME algorithm
        #############################################################################################
        #############################################################################################
        ##For the diabetes dataset:
        path_to_data="c:/HELIME/data_diabetes/"
        path_to_models="c:/HELIME/models/"
        path_to_results="c:/HELIME/results_diabetes/"
           
       
        
        df_all = pd.read_csv(path_to_data+"diabetes_data.csv")



        maxim_all=df_all.max(axis=0,numeric_only=True) # will return max value of each column
        minim_all=df_all.min(axis=0,numeric_only=True) # will return max value of each column
        #print("maxim_all=",maxim_all)
        #print("minim_all=",minim_all)
        
        
        #build the correlation matrix for train set
        #https://www.easycalculation.com/statistics/learn-correlation-matrix.php

        df_py = np.load(path_to_data+"X_train.npy")
        np.savetxt(path_to_data+"X_train.csv", df_py, fmt='%f',delimiter=",")
        df = pd.read_csv(path_to_data+"X_train.csv")
        corr_matrix=df.corr()  
        
        #save the corr_matrix
        
        corr_matrix.to_csv(path_to_results+"corr_matrix.csv")
        
        

        #Compute the Cholesky matrix of corr_matrix
        c = cholesky(corr_matrix, lower=True)
        
        #save Cholesky matrix
        np.savetxt(path_to_results+"c.csv", c, fmt='%f',delimiter=",")
        #print(c)
        

        

        ###############################################################################
        #create data, inverse:
        data_row=data_row
        #print("data_row=",data_row)
        no_features=data_row.shape[0]
                    
        if len(categorical_features)==0:
            no_categorical=0
        else:
            no_categorical=len(categorical_features)
       
               
        n_radial=n_radial              
         
        n_fi=n_fi                
        h_fi=(math.pi)/(n_fi)      
        h_radial=1/n_radial
        
        Pr=1
        
        no_features_to_perturb=no_features-no_categorical  
        
        # the instance x to explain is data_row
        instance=list(data_row)
        
        inverse=np.array(instance)
        alfa=np.zeros(no_features_to_perturb)
        for i in range(no_features_to_perturb):
                alfa[i]=min(instance[i]-minim_all[i],maxim_all[i]-instance[i]) 
        
        alfa_max=Pr*alfa
        for punct in range(int(num_samples/2)):
            newpoint1=list(instance)   
            newpoint2=list(instance)       
            j_radial=random.randint(1,n_radial)            
                           
            j_fi=[random.randint(n_fi/3,2*n_fi/3) for i in range(1, no_features_to_perturb+1)] 
                        
            j_fi[no_features_to_perturb-1]=random.randint(0,2*n_fi-1)   
            
            base=1
            
            
            for i in range(1,no_features_to_perturb): 
                    
                    newpoint1[no_features_to_perturb-i]=instance[no_features_to_perturb-i]+(j_radial*h_radial)*alfa_max[no_features_to_perturb-i]*base*math.cos((j_fi[i]*h_fi))
                    newpoint2[no_features_to_perturb-i]=instance[no_features_to_perturb-i]-(j_radial*h_radial)*alfa_max[no_features_to_perturb-i]*base*math.cos((j_fi[i]*h_fi))
                    base=base*math.sin((j_fi[i]*h_fi))
                
              
            newpoint1[0]=instance[0]+(j_radial*h_radial)*alfa_max[0]*base
            newpoint2[0]=instance[0]-(j_radial*h_radial)*alfa_max[0]*base  
            
            inverse = np.vstack((inverse,(newpoint1)))
                 
            
            inverse = np.vstack((inverse,(newpoint2)))    
            
            
         
        # ########################## for x*     
        instance=list(x_star)

        inverse = np.vstack((inverse,(x_star)))
        alfa=np.zeros(no_features_to_perturb)
        for i in range(no_features_to_perturb):
            alfa[i]=min(instance[i]-minim_all[i],maxim_all[i]-instance[i]) 
        
        alfa_max=Pr*alfa
        
        for punct in range(int(num_samples/2)):
              newpoint1=list(instance)
              newpoint2=list(instance)
              j_radial=random.randint(1,n_radial)

                       
              j_fi=[random.randint(n_fi/3,2*n_fi/3) for i in range(1, no_features_to_perturb+1)] 
             
              j_fi[no_features_to_perturb-1]=random.randint(0,2*n_fi-1)   
            
              base=1
              
             
              for i in range(1,no_features_to_perturb): 
                    
                     newpoint1[no_features_to_perturb-i]=instance[no_features_to_perturb-i]+(j_radial*h_radial)*alfa_max[no_features_to_perturb-i]*base*math.cos((j_fi[i]*h_fi))
                     newpoint2[no_features_to_perturb-i]=instance[no_features_to_perturb-i]-(j_radial*h_radial)*alfa_max[no_features_to_perturb-i]*base*math.cos((j_fi[i]*h_fi))
                     base=base*math.sin((j_fi[i]*h_fi))
                
              
              newpoint1[0]=instance[0]+(j_radial*h_radial)*alfa_max[0]*base
              newpoint2[0]=instance[0]-(j_radial*h_radial)*alfa_max[0]*base  
              
              inverse = np.vstack((inverse,(newpoint1)))
              
              inverse = np.vstack((inverse,(newpoint2)))    
             
        
        
        
        ##Compute the predictions for each point from the new dataset $V:=inverse$ and build the predictions vector yss
        yss = predict_fn(inverse) 
        #compute the mean for the each feature on the train dataset:
        
        mean_vector=np.zeros(no_features)
        
        for i in range(no_features):
            mean_vector[i]=np.mean(self.training_data[:,i])

            
            
        ###Replace null values from $V$ with $\overline {p}\cdot mean\_ value$, mean\_ value:=mean_vector[i]
       
        
        no_rows=inverse.shape[0]
        no_columns=inverse.shape[1] 
        te=0
        neg=0
        overline_p=1/100
        for i in range(no_rows):
            for j in range(no_columns):
               if inverse[i,j]==0:
                    te=te+1
                    inverse[i,j]=overline_p*mean_vector[j]
               if inverse[i,j]<0:
                    neg=neg+1
                    
        print("The number of replaced values=",te) 
        print("The number of negatives values=",neg)
         
       
        
        all_points=inverse   
        print(np.std(all_points[1,:]))
        #
        # Transform data from $V:=all_points$ into a normal distributed set of data, $V_{nd}:=all_points_nd$             
        all_points_1d, lam = boxcox(all_points.ravel())
        #print("lam=", lam)
        #print("shape results1d",result1d.shape)
        all_points_nd = np.transpose(all_points_1d.reshape(all_points.shape))
        #print("all_points_nd=",all_points_nd)
        
        #Compute $V_{final}=(C\cdot V_{nd})^{T},$ where $V_{final}:=inverse$ is the new dataset used for training the interpretable model. 
         
        c_all_points=np.dot(c,all_points_nd)
              
        inverse=np.transpose(c_all_points)
        
        ##################################################
        ###########end of HE-LIME algorithm

        # For the categorical features the original LIME algorithm is applied:
        
        for column in categorical_features:
            values = self.feature_values[column]
            freqs = self.feature_frequencies[column]
            inverse_column = self.random_state.choice(values, size=2*num_samples+2,
                                                      replace=True, p=freqs)
            
            inverse_column[0] = data_row[column]            
            inverse[:, column] = inverse_column

        #print("inverse=",inverse)
        # data is not used in HE-LIME algorithm
        no_rows=inverse.shape[0]
        no_columns=inverse.shape[1] 
        data = np.zeros((no_rows, no_columns))
        

        first_row_quartille = self.discretizer.discretize(data_row)

        for i in range(0,no_rows):
             row=self.discretizer.discretize(inverse[i,:])
             for j in range(no_columns):
                  if row[j] ==first_row_quartille[j]:
                     data[i,j]=1                  
                      
                  else:
                      data[i,j]=0 
        

        
        self.scaler.fit(inverse)
        
        
        scaled_data = (inverse- self.scaler.mean_) / self.scaler.scale_

        #Observation: No need the procedure:  __data_inverse(self, data_row, num_samples,  sampling_method)
        #This is replaced with the algorithm HE-LIME presented above
        #########################################################################################
        ########################################################################################
        
        distances = sklearn.metrics.pairwise_distances(
                scaled_data,
                scaled_data[0].reshape(1, -1),
                metric=distance_metric
                ).ravel()

        
        no=0
        for ni in range(yss.shape[0]):
             if yss[ni,0]>0.5:
                 no=no+1
        print("New instances in class 0:", no)         
        
        if self.mode == "classification":
            if len(yss.shape) == 1:
                raise NotImplementedError("HE-LIME does not currently support "
                                          "classifier models without probability "
                                          "scores. If this conflicts with your "
                                          "use case, please let us know: "
                                          "https://github.com/datascienceinc/lime/issues/16")
            elif len(yss.shape) == 2:
                if self.class_names is None:
                    self.class_names = [str(x) for x in range(yss[0].shape[0])]
                else:
                    self.class_names = list(self.class_names)
                if not np.allclose(yss.sum(axis=1), 1.0):
                    warnings.warn("""
                    Prediction probabilties do not sum to 1, and
                    thus does not constitute a probability space.
                    Check that you classifier outputs probabilities
                    (Not log probabilities, or actual class predictions).
                    """)
            else:
                raise ValueError("Your model outputs "
                                 "arrays with {} dimensions".format(len(yss.shape)))

        else:
            try:
                assert isinstance(yss, np.ndarray) and len(yss.shape) == 1
            except AssertionError:
                raise ValueError("Your model needs to output single-dimensional \
                    numpyarrays, not arrays of {} dimensions".format(yss.shape))

            predicted_value = yss[0]
            min_y = min(yss)
            max_y = max(yss)

            yss = yss[:, np.newaxis]

        feature_names = copy.deepcopy(self.feature_names)
        if feature_names is None:
            feature_names = [str(x) for x in range(data_row.shape[0])]

        values = self.convert_and_round(data_row)

        for i in self.categorical_features:
            if self.discretizer is not None and i in self.discretizer.lambdas:
                continue
            name = int(data_row[i])
            if i in self.categorical_names:
                name = self.categorical_names[i][name]
            feature_names[i] = '%s=%s' % (feature_names[i], name)
            values[i] = 'True'
        categorical_features = self.categorical_features

        discretized_feature_names = None
        if self.discretizer is not None:
            categorical_features = range(data.shape[1])
            discretized_instance = self.discretizer.discretize(data_row)
            discretized_feature_names = copy.deepcopy(feature_names)
            for f in self.discretizer.names:
                discretized_feature_names[f] = self.discretizer.names[f][int(
                        discretized_instance[f])]

        domain_mapper = TableDomainMapper(feature_names,
                                          values,
                                          scaled_data[0],
                                          categorical_features=categorical_features,
                                          discretized_feature_names=discretized_feature_names)
        ret_exp = explanation.Explanation(domain_mapper,
                                          mode=self.mode,
                                          class_names=self.class_names)
        ret_exp.scaled_data = scaled_data
        if self.mode == "classification":
            ret_exp.predict_proba = yss[0]
            if top_labels:
                labels = np.argsort(yss[0])[-top_labels:]
                ret_exp.top_labels = list(labels)
                ret_exp.top_labels.reverse()
        else:
            ret_exp.predicted_value = predicted_value
            ret_exp.min_value = min_y
            ret_exp.max_value = max_y
            labels = [0]

        
        if (yss[0][0]>0.5):
            labels=[0]
        else:
            labels=[1]    
        
        ##choose the explanation for the label of the instance to be explain
        for label in labels:  
            (ret_exp.intercept[label],
             ret_exp.local_exp[label],
             ret_exp.score, ret_exp.local_pred) = self.base.explain_instance_with_data(
                    scaled_data,
                    yss,
                    distances,
                    label,
                    num_features,
                    model_regressor=model_regressor,
                    feature_selection=self.feature_selection)
            
      
        if self.mode == "regression":
            
            ret_exp.intercept[1] = ret_exp.intercept[0]
            ret_exp.local_exp[1] = [x for x in ret_exp.local_exp[0]]
            ret_exp.local_exp[0] = [(i, -1 * j) for i, j in ret_exp.local_exp[1]]
        
        return ret_exp