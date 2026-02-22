# Python_ML_MLPClassifier
This project is a classification case based on traffic accident data, which includes data cleaning, feature engineering, training and parameter tuning of multiple models (Baseline MLP, Tuned MLP, Bagging MLP, Voting Ensemble), and comparing and ranking the above four types of models according to metrics such as accuracy, precision, recall, F1-Score, and AUC.

# Process
1.	Prepare data (remove duplicates, clean the data, handle missing values, fix data types).
2.	Encode categorical variables: using Label Encoding for ordered categories and One-Hot Encoding for unordered categories.
3.	Split the set into training and testing size (80%/20% split) to prevent any future data leakage.
4.	Perform feature selection on the training set, assess the importance of different features using a feature importance plot, then apply the same selected features to the test set.
5.	Perform feature scaling on the numerical features: fit the scaler on the training set, then transform the test set using the same parameters to avoid data leakage.
6.	Train the model using the processed training data.
7.	Fine-tune model using RandomizedSearchCV.
8.	Explore two ensemble methods: Voting Classifier+Stacking Classifier
9.	Evaluation metrics such as accuracy, precision, recall, F1-score, and ROC-AUC


# Clean data

# Select feature

<img width="1189" height="790" alt="image" src="https://github.com/user-attachments/assets/3cb3138b-5857-4bda-b727-1945e90718e3" />

Final 24 features selected:

 1. Police_Force_label            
 2. Casualties_Per_Vehicle        
 3. Number_of_Vehicles            
 4. Day_Cos                       
 5. Day_Sin                       
 6. Hour_Cos                      
 7. Hour_Sin                      
 8. Month_Sin                     
 9. Speed_Category_Low_on         
10. Day_of_Week_label             
11. Geo_Cluster                   
12. Vehicle_Type_label            
13. Road_Type_Single carriageway_on
14. Month_Cos                     
15. Light_Conditions_Daylight_on  
16. Weather_Conditions_Fine no high winds_on
17. Year                          
18. Junction_Control_Give way or uncontrolled_on
19. Junction_Control_Data missing or out of range_on
20. Carriageway_Hazards_IsNone_on 
21. Road_Surface_Conditions_Wet or damp_on
22. Junction_Detail_Not at junction or within 20 metres_on
23. Junction_Detail_Roundabout_on 
24. Light_Conditions_Darkness - no lighting_on

# Train Model and compare multi methods
<img width="1789" height="985" alt="image" src="https://github.com/user-attachments/assets/bebba069-1d6f-4812-88c2-bc9447d954fb" />
<img width="928" height="589" alt="image" src="https://github.com/user-attachments/assets/373b11d3-b1eb-4dbe-bd79-9b855388274f" />
<img width="978" height="1546" alt="image" src="https://github.com/user-attachments/assets/689945c1-cc34-4230-80a4-51f19812e76d" />

The best model is the Baseline MLP, MLPClassifier.


