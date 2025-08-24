import pandas as pd

def feature_engineering(df:pd.DataFrame):
    try : 
        df["work_type"] = df["workclass"].replace({
        "Self-emp-not-inc": "Self-Employed",
        "Self-emp-inc": "Self-Employed",
        "Local-gov" : "Gov_employee",
        "State_gov" : "Gov_employee",
        "Without-pay": "Other",
        "Never-worked": "Other"
        })

        df["is_married"] = df["marital-status"].apply(lambda x: 1 if "Married" in x else 0)

        def group_country(country):
            if country in ['United-States','Mexico','Canada']:
                return 'North America'
            elif country in ['India', 'China', 'Philippines', 'Japan','Taiwan','Iran']:
                return 'Asia'
            elif country in ['Germany', 'England', 'France', 'Italy','Poland']:
                return 'Europe'
            else:
                return 'Other'

        df['region'] = df['native-country'].apply(group_country)
        df["race_gender"] = df["race"] + "_" + df["gender"]

        df["has_capital_gain"] = (df["capital-gain"] > 0).astype(int)
        df["has_capital_loss"] = (df["capital-loss"] > 0).astype(int)
        
        df = df.drop(columns=['race','gender','native-country','marital-status','workclass'],axis=1)
        df = df.drop('education',axis=1)

        return df
    
    except Exception as e:
        print(f'An unexpected error occured during feature_engineering : {e}') 
        raise