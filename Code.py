import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from scipy.stats import chi2_contingency
from sklearn.svm import SVC


df= pd.read_csv("/content/Dataset#2 (2).csv")
df.head()


#EDA Stage:
new_feature_names = [
    'Age', 'Gender', 'Thyroid_Hormone_Medication', 'Thyroid_Surgery', 'Thyroid_Radiation',
    'Thyroid_Status', 'Nodule_Type', 'Precancerous_Lesion', 'Tumor_Type', 'Tumor_Focus',
    'Tumor_Size', 'T (Tumor)', 'N (Nodes)', 'M (Metastasis)',
    'Cancer Staging', 'Response to Initial Therapy', 'DX'
]


df.columns = new_feature_names


df.shape
df.info()
df.duplicated().sum()
df= df.drop_duplicates()
df.shape
df2= df.copy()
df2= df2.reset_index(drop=True)



#Relationships and visualization each category:
categorical_features = ['Gender', 'Thyroid_Hormone_Medication', 'Thyroid_Surgery', 'Thyroid_Radiation',
                        'Thyroid_Status', 'Nodule_Type', 'Precancerous_Lesion',
                        'Tumor_Type', 'Tumor_Focus', 'T (Tumor)', 'N (Nodes)', 'M (Metastasis)',
                         'Cancer Staging', 'Response to Initial Therapy', 'DX']

for feature in categorical_features:

    sns.countplot(x=feature, data=df2)
    plt.title(f' count plot for {feature}')
    plt.show()

#--------------------------------------------------

for feature in categorical_features:

    sns.countplot(x=feature, hue='DX', data=df2)
    plt.title(f'{feature} vs. DX')
    plt.show()

    # Perform Chi-Square test to show a stringth of a relationship.
    contingency_table = pd.crosstab(df2[feature], df2['DX'])
    chi2, p, _, _ = chi2_contingency(contingency_table)

    print(f"{feature} - Chi-Square Value: {chi2}, p-value: {p}")



sns.boxplot(x='DX', y='Age', data=df2)
plt.title('Age vs. DX (Box Plot)')
plt.show()  #potential relationship based on the difference of the median.

df2['DX'].value_counts()


df2['Gender'].value_counts()
df2['Thyroid_Hormone_Medication'].value_counts()
df2['Thyroid_Surgery'].value_counts()
df2['Thyroid_Radiation'].value_counts()
df2['Thyroid_Status'].value_counts()
df2['Nodule_Type'].value_counts()
df2['Precancerous_Lesion'].value_counts()
df2['Tumor_Type'].value_counts()
df2['Tumor_Focus'].value_counts()
df2['Tumor_Size'].value_counts()
df2['T (Tumor)'].value_counts()
df2['N (Nodes)'].value_counts()
df2['M (Metastasis)'].value_counts()
df2['Cancer Staging'].value_counts() #The symbols I, II, III, IV, etc., typically represent stages in cancer classification, commonly referred to as cancer staging.
df2['Response to Initial Therapy'].value_counts()


#T (Tumor): Describes the size of the primary tumor and whether it has invaded nearby tissues.
#N (Nodes): Indicates whether the cancer has spread to nearby lymph nodes.
#M (Metastasis): Denotes whether the cancer has metastasized (spread) to other parts of the body.



df3 = df2.drop(['Thyroid_Radiation',
    'Tumor_Type',
    'Nodule_Type',
    'Thyroid_Surgery',
    'Thyroid_Status',
    'M (Metastasis)',
    'Thyroid_Radiation',
    'Thyroid_Status'], axis= 1).copy()


label_columns = ['DX', 'Gender', 'Thyroid_Hormone_Medication']
label_encoder = LabelEncoder()

for column in label_columns:
    df3[column] = label_encoder.fit_transform(df3[column])


columns_to_one_hot_encode = df3.select_dtypes(include=['object']).columns.difference(label_columns)
one_hot_encoder = OneHotEncoder(drop='first', sparse=False)

for column in columns_to_one_hot_encode:
    encoded_data = one_hot_encoder.fit_transform(df3[[column]])
    new_columns = [f"{column}_{category}" for category in one_hot_encoder.get_feature_names_out([column])]

    df3 = pd.concat([df3, pd.DataFrame(encoded_data, columns=new_columns)], axis=1)
    df3.drop(column, axis=1, inplace=True)





#--------------------------------------------------------------------

#Here we moved to he ML Stage to make a prediction and so on:


X = df3.drop(['DX'], axis=1)
y = df3['DX']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

classifier = RandomForestClassifier(n_estimators=100, random_state=42)


classifier.fit(X_train, y_train)


y_val_pred = classifier.predict(X_val)


accuracy_val = accuracy_score(y_val, y_val_pred)
print("Validation Accuracy:", accuracy_val)


y_test_pred = classifier.predict(X_test)


accuracy_test = accuracy_score(y_test, y_test_pred)
print("Test Accuracy:", accuracy_test)


#-----------------------------------------------------------


X = df3.drop(['DX'], axis=1)
y = df3['DX']


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=42)
X_test, y_test = X_val, y_val

model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dropout(0.2),  # Dropout layer to prevent overfitting
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),  # Dropout layer to prevent overfitting
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=25, batch_size=28, validation_data=(X_val, y_val))

y_pred = model.predict(X_test)
y_pred_binary = [1 if pred > 0.5 else 0 for pred in y_pred]

accuracy = accuracy_score(y_test, y_pred_binary)

print("Accuracy:", accuracy)



#--------------------------------------------------------------------------


X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly'],
    'degree': [2, 3, 4],
    'gamma': ['scale', 'auto'],
}

svm_classifier = SVC(random_state=42)

grid_search = GridSearchCV(svm_classifier, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

best_svm_classifier = grid_search.best_estimator_


y_val_pred = best_svm_classifier.predict(X_val)
accuracy_val = accuracy_score(y_val, y_val_pred)



y_test_pred = best_svm_classifier.predict(X_test)
accuracy_test = accuracy_score(y_test, y_test_pred)



print('\nValidation Accuracy: ', accuracy_val)
print('\nTest Accuracy: ', accuracy_test)


