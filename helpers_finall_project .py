import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder


def grab_col_names(dataframe, cat_th=10, car_th=20, excluded=None):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """
    excluded = [] if excluded is None else excluded
    df_cols = [i for i in dataframe.columns if i not in excluded]

    num_but_cat = [col for col in df_cols if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in df_cols if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    #cat_cols = cat_cols + num_but_cat

    # cat_cols
    cat_cols = [col for col in df_cols if dataframe[col].dtypes == "O"]
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in df_cols if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    # print(f"Observations: {dataframe.shape[0]}")
    # print(f"Variables: {dataframe.shape[1]}")
    # print(f'cat_cols: {len(cat_cols)}')
    # print(f'num_cols: {len(num_cols)}')
    # print(f'cat_but_car: {len(cat_but_car)}')
    # print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car, num_but_cat
def plot_importance(model, features, num=20, save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    feature_imp = feature_imp.sort_values(by="Value", ascending=False).reset_index(drop=True)
    new_features_importance_order = feature_imp[feature_imp['Feature'].str.contains("NEW_")]
    feature_imp_print = feature_imp[0:num]
    new_features_importance_order = new_features_importance_order[0:num]
    print(feature_imp_print, "\n")
    print(new_features_importance_order)

    sns.barplot(x="Value", y="Feature", data=feature_imp_print)
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')
    res = feature_imp_print['Feature'].values.tolist()
    return res
def get_col_desc(name='SK_ID_CURR', col='application_{train|test}.csv'):
    desc_df = pd.read_csv("datasets/home-credit-default-risk/HomeCredit_columns_description.csv", engine='python')
    desc_df.style.set_properties(subset=[name], **{'text-align': 'left'})
    print(desc_df[(desc_df["Table"] == col) & (desc_df["Row"] == name)][["Row", 'Special', 'Description']])
def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(dropna=False),
                            "RATIO": dataframe[col].value_counts(dropna=False) / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")
def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                    and (temp_df[col].value_counts(dropna=False) / len(temp_df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts(dropna=False) / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

    return temp_df
def target_summary_with_cat_extended(dataframe, target, categorical_col):
    """

        Veri setindeki Target colonunu, girilen kategorik colona gore gruplayip
            - mean
            - count
            - ratio
        sonuclarini ekrana yazdirir.

        Parameters
        ------
            dataframe: dataframe
                    Target ve Kategorik kolonlarin bulundugu dataframe
            target: str
                    Sonucun getirilecegi hedef degisken
            categorical_col: str
                    Gruplanmak istenen kategorik kolon

        Returns
        ------
            None

        Examples
        ------
            import pandas as pd
            in:
            df = pd.DataFrame({'Animal': ['Falcon', 'Falcon', 'Falcon',
                                          'Parrot', 'Parrot'],
                               'Max Speed': [310, 330, 340, 24, 28]})

            in:
            df
            out:
              Animal   Max Speed
            0  Falcon  310
            1  Falcon  330
            2  Falcon  340
            3  Parrot   24
            4  Parrot   28

            in: target_summary_with_cat_extended(df, 'Max Speed', 'Animal')
            out:
                    TARGET_MEAN  TARGET_CCOUNT  RATIO
            Animal
            Falcon 326.6667      3             60.0000
            Parrot  26.0000      2             40.0000

        Notes
        ------
            None

        """

    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean(),
                        "TARGET_CCOUNT": dataframe.groupby(categorical_col)[target].count(),
                        "RATIO": 100 * dataframe[categorical_col].value_counts() / len(dataframe)}), end="\n\n\n")
def replace_with_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name, q1, q3)
    dataframe.loc[(dataframe[col_name] < low_limit), col_name] = low_limit
    dataframe.loc[(dataframe[col_name] > up_limit), col_name] = up_limit
def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe
def grab_col_names_c(dataframe, cat_th=10, car_th=20, ignore_vars=[]):
    # excluded columns
    exc_cols = []
    if type(ignore_vars) is not list:
        exc_cols.append(ignore_vars)
    else:
        exc_cols.extend(ignore_vars)
        # for i in ignore_vars:
        #    exc_cols.append(i)
    #print(exc_cols)

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]

    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]

    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car and col not in exc_cols]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat and col not in exc_cols]

    # print(f"Observations: {dataframe.shape[0]}")
    # print(f"Variables: {dataframe.shape[1]}")
    # print(f'cat_cols: {len(cat_cols)}:', cat_cols)
    # print(f'num_cols: {len(num_cols)}:', num_cols)
    # print(f'cat_but_car: {len(cat_but_car)}:', cat_but_car)
    # print(f'num_but_cat: {len(num_but_cat)}:', num_but_cat)
    # print(f'excluded_cols: {len(exc_cols)}:', exc_cols)
    # print("Print >> cat_cols >> num_cols >> cat_but_car >> exc_cols:")
    return cat_cols, num_cols, cat_but_car, exc_cols
def one_hot_encoder_c(dataframe, categorical_cols, drop_first=False, nan_as_category=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first, dummy_na=nan_as_category)
    return dataframe
def outlier_thresholds_c(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit
def cat_summary_with_target(dataframe, col_name, target, plot=False):
    sum_df = pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe),
                        "Target Mean": dataframe.groupby(col_name)[target].mean()})
    if dataframe[col_name].isnull().sum() > 0:
        nan_df = pd.DataFrame({col_name: dataframe[col_name].isnull().sum(),
                               "Ratio": 100 * dataframe[col_name].isnull().sum() / dataframe.shape[0],
                               "Target Mean": np.nan}, index=[np.nan])
        sum_df = sum_df.append(nan_df)
    print(sum_df, end="\n")

    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()
def one_hot_encoder_z(df, nan_as_category=True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns=categorical_columns, dummy_na=nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns


