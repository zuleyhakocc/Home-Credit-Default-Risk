###########################################################################################################
# Home Credit Default Risk
###########################################################################################################


# Main
import gc
import time
from contextlib import contextmanager
import re

import catboost.core
import lightgbm.sklearn
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import missingno as msno
import pickle
import joblib

# Visuals
from yellowbrick.classifier import ROCAUC, PrecisionRecallCurve, confusion_matrix, ClassificationReport
from yellowbrick.contrib.wrapper import wrap
from yellowbrick.model_selection import FeatureImportances

# ML 1
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier, \
    RandomForestClassifier, HistGradientBoostingClassifier, StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier

# EDA
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import RobustScaler, MinMaxScaler, LabelEncoder, StandardScaler

# Evaluation Libraries
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, cross_validate, \
    train_test_split, validation_curve
from sklearn.metrics import accuracy_score, auc, classification_report, confusion_matrix, f1_score, recall_score, \
    roc_auc_score, roc_curve, precision_score, plot_roc_curve

# ML 2
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from HAFTA_FINAL_PROJECT.helpers_final_project import *

# Notebook Settings
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")

pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', 1000)
pd.set_option('display.max_rows', 50)
pd.set_option('display.width', 50000)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.4f' % x)
pd.set_option('display.colheader_justify', 'left')


# -- Utilties
@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - finished in {:.0f}s".format(title, time.time() - t0))


def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    # Memory usage of dataframe is 1702.81 MB
    # Memory usage after optimization is: 442.01 MB
    # Decreased by 74.0 %
    return df


# -- EDA
def eda_base():
    """
    runs a base eda to df in order to compare the effects of feature engineering etc. on model
    Returns
    -------
    model ready df.
    """
    ######################################
    # Missing Values
    ######################################
    # cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(df)
    # Observations: 356255
    # Variables: 122
    # cat_cols: 15
    # num_cols: 67
    # cat_but_car: 1
    # num_but_cat: 39
    global train, test, df
    train = pd.read_csv('datasets/home-credit-default-risk/application_train.csv')
    test = pd.read_csv('datasets/home-credit-default-risk/application_test.csv')
    df = train.append(test).reset_index(drop=True)

    df.isnull().sum()
    df.isnull().sum().sum()  # 10670198
    df.shape
    # df.dropna(inplace=True)
    # msno.matrix(df.sample(250))
    # plt.show()

    df = df[df['CODE_GENDER'] != 'XNA']
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)

    df[df.columns[df.isnull().any()]]

    cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(df)

    na_cols_cat = [col for col in cat_cols if df[col].isnull().sum() > 0]
    df[na_cols_cat] = df[na_cols_cat].apply(lambda x: x.fillna(x.mode()), axis=0)

    na_cols_num = [col for col in num_cols if df[col].isnull().sum() > 0 and "TARGET" not in col]
    df[na_cols_num] = df[na_cols_num].apply(lambda x: x.fillna(x.median()), axis=0)

    na_cols_cat_but_car = [col for col in cat_but_car if df[col].isnull().sum() > 0]
    df[na_cols_cat_but_car] = df[na_cols_cat_but_car].apply(lambda x: x.fillna(x.mode()), axis=0)

    na_cols_num_but_cat = [col for col in num_but_cat if df[col].isnull().sum() > 0 and "TARGET" not in col]
    df[na_cols_num_but_cat] = df[na_cols_num_but_cat].apply(lambda x: x.fillna(x.median()), axis=0)

    df['OCCUPATION_TYPE'] = df['OCCUPATION_TYPE'].fillna(df['OCCUPATION_TYPE'].mode()[0])

    ######################################
    # Feature Engineering
    ######################################

    #############################################
    # Outliers
    #############################################

    #############################################
    # Label Encoding
    #############################################

    #############################################
    # Rare Encoding
    #############################################

    #############################################
    # One-Hot Encoding
    #############################################
    df = pd.get_dummies(df, dummy_na=True)
    df.shape
    #############################################
    # Standart Scaler
    #############################################

    ######################################
    # Modeling
    ######################################
    global train_df, test_df
    train_df = df[df['TARGET'].notnull()]
    test_df = df[df['TARGET'].isnull()].drop("TARGET", axis=1)

    global X, y, X_train, X_test, y_train, y_test
    y = train_df["TARGET"]
    X = train_df.drop(["SK_ID_CURR", "TARGET"], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)


def eda_application():
    ######################################
    # 1. Read Data
    ######################################
    # cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(df)
    # Observations: 356255
    # Variables: 122
    # cat_cols: 15
    # num_cols: 67
    # cat_but_car: 1
    # num_but_cat: 39

    train = pd.read_csv('datasets/home-credit-default-risk/application_train.csv')
    test = pd.read_csv('datasets/home-credit-default-risk/application_test.csv')
    df = train.append(test).reset_index(drop=True)

    del train, test
    gc.collect()

    ######################################
    # 2. Missing Values & Preprocessing
    ######################################
    df.isnull().sum().sort_values(ascending=False).head(20)  # 248360 and so on...
    df.isnull().sum().sum()  # 10670198 total of cell that is null
    df[df.columns[df.isnull().any()]]  # 68 columns that has nulls in it
    df.shape  # (356255, 122)
    # df.dropna(inplace=True)
    # msno.matrix(df.sample(250)) # some columns have same patterns for NANs as expected.
    # plt.show()

    # 4 row deleted.
    df = df[df['CODE_GENDER'] != 'XNA']

    # at 64648 rows, 365243 values are at NAN occupied.
    df[df['DAYS_EMPLOYED'] == 365243]['OCCUPATION_TYPE'].value_counts(dropna=False)
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)

    # has 0 child
    df[df['CNT_FAM_MEMBERS'].isnull()]['CNT_CHILDREN']
    # 2 rows filled with 0
    df['CNT_FAM_MEMBERS'].fillna(0, inplace=True)

    # Change 0 to null
    df['DAYS_LAST_PHONE_CHANGE'].replace(0, np.nan, inplace=True)

    # # Checking 0 values at df
    # lst = []
    # for col in df.columns:
    #     try:
    #         if df[col].min()==0 and df[col].nunique()!=2:
    #             print(col)
    #             lst.append(col)
    #     except:
    #         pass
    #
    # for col in lst:
    #     print(col, len(df[df[col]==0][col]))

    # -----------------------------

    # For basic imputing we will fill values with mode or median.
    # cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(df)
    # na_cols_cat = [col for col in cat_cols if df[col].isnull().sum() > 0]
    # df[na_cols_cat] = df[na_cols_cat].apply(lambda x: x.fillna(x.mode()[0]), axis=0)
    # na_cols_num = [col for col in num_cols if df[col].isnull().sum() > 0 and "TARGET" not in col]
    # df[na_cols_num] = df[na_cols_num].apply(lambda x: x.fillna(x.median()), axis=0)
    # na_cols_cat_but_car = [col for col in cat_but_car if df[col].isnull().sum() > 0]
    # df[na_cols_cat_but_car] = df[na_cols_cat_but_car].apply(lambda x: x.fillna(x.mode()), axis=0)
    # na_cols_num_but_cat = [col for col in num_but_cat if df[col].isnull().sum() > 0 and "TARGET" not in col]
    # df[na_cols_num_but_cat] = df[na_cols_num_but_cat].apply(lambda x: x.fillna(x.median()), axis=0)

    # Only TARGET 48744 left which is from test csv.
    df.isnull().sum().sort_values(ascending=False)

    # "yes" and "no" change to 0-1
    cols = ['FLAG_OWN_CAR', 'EMERGENCYSTATE_MODE', 'FLAG_OWN_REALTY']
    for col in cols:
        df[col] = df[col].apply(lambda x: x if x is np.nan else (0 if 'N' in x else 1))

    ######################################
    # 3.1 Feature Engineering
    ######################################
    # child count over age
    df['NEW_AGE_OVER_CHLD'] = df['CNT_CHILDREN'] / (df['DAYS_BIRTH'] / 365)

    df['NEW_LIVEAREA_OVER_FAM_MEMBERS'] = df['LIVINGAREA_AVG'] / (df['CNT_FAM_MEMBERS'] + 1)

    # How many documents is OK out of total docs
    cols_to_sum = [col for col in df.columns if "FLAG_DOCUMEN" in col]
    df['NEW_DOCUMENT_FULLFILLMENT'] = df[cols_to_sum].sum(axis=1) / len(cols_to_sum)

    # How many items does client has
    cols_to_sum = [col for col in df.columns if "FLAG_" in col]
    cols_to_sum = [col for col in cols_to_sum if "FLAG_DOC" not in col]
    df['NEW_FLAG_ITEMS'] = df[cols_to_sum].sum(axis=1) / len(cols_to_sum)

    # Proportion of days worked and age
    df['NEW_AGE_OVER_WORK'] = df['DAYS_EMPLOYED'] / (df['DAYS_BIRTH'] / 365)

    # Proportion of income total over credit applied
    df['NEW_INCOME_OVER_CREDIT'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']

    # Proportion of credit income over credit applied
    df['NEW_ANNUITY_OVER_CREDIT'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']

    # Property ownage
    df['NEW_PROPERTY_TOTAL'] = 0.1 * df['FLAG_OWN_CAR'] + 0.9 * df['FLAG_OWN_REALTY']

    # Proportion of credir applied over the GOODS that he/she will purchase
    df['NEW_PROPERTY_TOTAL'] = df['AMT_GOODS_PRICE'] + df['AMT_CREDIT']

    # Number of enquiries to Credit Bureau about the client weightage
    df['NEW_MT_REQ_CREDIT_BUREAU'] = 0.3 * df['AMT_REQ_CREDIT_BUREAU_HOUR'] + 0.25 * df['AMT_REQ_CREDIT_BUREAU_DAY'] + \
                                     0.175 * df['AMT_REQ_CREDIT_BUREAU_WEEK'] + 0.125 * df[
                                         'AMT_REQ_CREDIT_BUREAU_MON'] + \
                                     0.1 * df['AMT_REQ_CREDIT_BUREAU_QRT'] + 0.05 * df['AMT_REQ_CREDIT_BUREAU_YEAR']

    # Home overall scoring
    df['NEW_HOME_OVERALL_SCORE'] = df['APARTMENTS_AVG'] * 5 + df['BASEMENTAREA_AVG'] * 2 + df['COMMONAREA_AVG'] * 4 + \
                                   df['ELEVATORS_AVG'] * 1 \
                                   + df['EMERGENCYSTATE_MODE'] * 1 + df['ENTRANCES_AVG'] * 1 + df['FLOORSMAX_AVG'] * 2 + \
                                   df['FLOORSMIN_AVG'] * 2 \
                                   + df['LANDAREA_AVG'] * 3 + df['LIVINGAPARTMENTS_AVG'] * 2 + df[
                                       'LIVINGAREA_AVG'] * 3 + \
                                   df['NONLIVINGAPARTMENTS_AVG'] * 1 + df['NONLIVINGAREA_AVG'] * 1 + df[
                                       'YEARS_BUILD_AVG'] * -1

    # EXT Source combinations
    df['NEW_EXT_1'] = df['EXT_SOURCE_2'] / df['EXT_SOURCE_3']
    df['NEW_EXT_2'] = df['EXT_SOURCE_2'] ** df['EXT_SOURCE_3']
    df['NEW_EXT_3'] = df['EXT_SOURCE_2'] / df['EXT_SOURCE_3'] * df['EXT_SOURCE_1']
    df['NEW_EXT_4'] = 2 * df['EXT_SOURCE_2'] + 3 * df['EXT_SOURCE_3'] + df['EXT_SOURCE_1']
    df['NEW_EXT_5'] = 4 * df['EXT_SOURCE_1'] + 2 * df['DAYS_BIRTH'] + 1 * df['AMT_ANNUITY'] + \
                      3 * df['EXT_SOURCE_2'] + 4 * df['AMT_GOODS_PRICE'] + 1.5 * df['DAYS_EMPLOYED']
    df['NEW_EXT_6'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']

    # --------------------------------------------------------------------------
    df['NEW_EXT1_TO_BIRTH_RATIO'] = df['EXT_SOURCE_1'] / (df['DAYS_BIRTH'] / 365)
    df['NEW_EXT3_TO_BIRTH_RATIO'] = df['EXT_SOURCE_3'] / (df['DAYS_BIRTH'] / 365)

    #############################################
    # 3.2 Rare Analyzing & Encoding
    #############################################
    # cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(df)
    # rare_analyser(df, "TARGET", cat_cols)
    rare_cols = ['NAME_INCOME_TYPE', 'NAME_TYPE_SUITE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS',
                 'NAME_HOUSING_TYPE', 'OCCUPATION_TYPE', 'WALLSMATERIAL_MODE', 'HOUSETYPE_MODE', 'ORGANIZATION_TYPE']
    for col in rare_cols:
        tmp = df[col].value_counts() / len(df)
        rare_labels = tmp[tmp < 0.05].index
        df[col] = np.where(df[col].isin(rare_labels), 'Rare', df[col])

    #############################################
    # 4. Outliers
    #############################################
    # I have tried different thresholds but everytime score decreased.
    # Tree models is not affected from outliers that much.
    # cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(df)
    # col_out = [col for col in num_cols if col not in ['TARGET', 'SK_ID_CURR'] and df[col].nunique() > 2]
    # for col in col_out:
    #     replace_with_thresholds(df, col, q1=0.05, q3=0.95)

    #############################################
    # 5. Label Encoding
    #############################################
    # cols that will change ['NAME_CONTRACT_TYPE', 'CODE_GENDER', 'NAME_HOUSING_TYPE', 'HOUSETYPE_MODE']
    binary_cols = [col for col in df.columns if df[col].dtype not in [int, float] and len(df[col].unique()) == 2]
    for col in binary_cols:
        label_encoder(df, col)

    #############################################
    # 6. Rare Encoding
    #############################################
    # Applied upper part

    #############################################
    # 7. One-Hot Encoding
    #############################################
    # df = pd.get_dummies(df, dummy_na=True)
    df = pd.get_dummies(df)
    # print("application shape: ", df.shape)

    #############################################
    # 8. Scaling (Best is with MinMax. Tried Robust and Standart too)
    #############################################
    col_sca = [col for col in df.columns if col not in ['TARGET', 'SK_ID_CURR'] and df[col].nunique() > 2]
    scaler = MinMaxScaler()
    df[col_sca] = scaler.fit_transform(df[col_sca])

    return df


def eda_bureau_bb():
    # bureau_balance tablosunun okutulması

    bb = pd.read_csv('datasets/bureau_balance.csv')

    bb = pd.get_dummies(bb)

    agg_list = {"MONTHS_BALANCE": "count",
                "STATUS_0": ["sum", "mean"],
                "STATUS_1": ["sum"],
                "STATUS_2": ["sum"],
                "STATUS_3": ["sum"],
                "STATUS_4": ["sum"],
                "STATUS_5": ["sum"],
                "STATUS_C": ["sum", "mean"],
                "STATUS_X": ["sum", "mean"]}

    bb_agg = bb.groupby("SK_ID_BUREAU").agg(agg_list)

    # Renaming the variable names
    bb_agg.columns = pd.Index([col[0] + "_" + col[1].upper() for col in bb_agg.columns.tolist()])

    # New features with the Status_sum's
    bb_agg['NEW_STATUS_SCORE'] = bb_agg['STATUS_1_SUM'] + bb_agg['STATUS_2_SUM'] ** 2 + bb_agg['STATUS_3_SUM'] ** 3 + \
                                 bb_agg['STATUS_4_SUM'] ** 4 + bb_agg['STATUS_5_SUM'] ** 5
    bb_agg['NEW_CRITICAL_STATUS'] = bb_agg['STATUS_3_SUM'] + bb_agg['STATUS_4_SUM'] * 2 + bb_agg['STATUS_5_SUM'] * 3

    bb_agg.drop(['STATUS_1_SUM', 'STATUS_2_SUM', 'STATUS_3_SUM', 'STATUS_4_SUM', 'STATUS_5_SUM'], axis=1, inplace=True)

    bureau = pd.read_csv('datasets/bureau.csv')
    bureau_and_bb = bureau.join(bb_agg, how='left', on='SK_ID_BUREAU')
    # Joining the BUREAU BALANCE and BUREAU

    # Reducing the number of classes of CREDIT_TYPE to 3
    bureau_and_bb['CREDIT_TYPE'] = bureau_and_bb['CREDIT_TYPE'].replace(['Car loan',
                                                                         'Mortgage',
                                                                         'Microloan',
                                                                         'Loan for business development',
                                                                         'Another type of loan',
                                                                         'Unknown type of loan',
                                                                         'Loan for working capital replenishment',
                                                                         "Loan for purchase of shares (margin lending)",
                                                                         'Cash loan (non-earmarked)',
                                                                         'Real estate loan',
                                                                         "Loan for the purchase of equipment",
                                                                         "Interbank credit",
                                                                         "Mobile operator loan"], 'Rare')

    # Reducing the number of classes of CREDIT_ACTIVE to 2
    bureau_and_bb['CREDIT_ACTIVE'] = bureau_and_bb['CREDIT_ACTIVE'].replace(['Bad debt', 'Sold'], 'Active')

    # 99% of the variable CREDIT_CURRENCY is currency1, we drop it.
    bureau_and_bb.drop(["SK_ID_BUREAU", "CREDIT_CURRENCY"], inplace=True, axis=1)

    bureau_and_bb = pd.get_dummies(bureau_and_bb, columns=["CREDIT_TYPE", "CREDIT_ACTIVE"])

    # NEW FEATURES
    agg_list = {
        "SK_ID_CURR": ["count"],
        "DAYS_CREDIT": ["min", "max", "mean", "var"],
        "CREDIT_DAY_OVERDUE": ["sum", "mean", "max"],
        "DAYS_CREDIT_ENDDATE": ["max", "min", "mean"],
        "DAYS_ENDDATE_FACT": ["max", "mean"],
        "AMT_CREDIT_MAX_OVERDUE": ["mean", "max"],
        "CNT_CREDIT_PROLONG": ["sum", "mean", "max", "min"],
        "AMT_CREDIT_SUM": ["sum", "mean", "max", "min"],
        "AMT_CREDIT_SUM_DEBT": ["sum", "mean", "max"],
        "AMT_CREDIT_SUM_LIMIT": ["sum", "mean", "max"],
        'AMT_CREDIT_SUM_OVERDUE': ["sum", "mean"],
        'DAYS_CREDIT_UPDATE': ["mean"],
        'AMT_ANNUITY': ["max", "mean"],
        'MONTHS_BALANCE_COUNT': ["sum"],
        'STATUS_0_SUM': ["sum"],
        'STATUS_0_MEAN': ["mean"],
        'STATUS_C_SUM': ["sum"],
        'STATUS_C_MEAN': ["mean"],
        'CREDIT_ACTIVE_Active': ["sum", "mean"],
        'CREDIT_ACTIVE_Closed': ["sum", "mean"],
        'CREDIT_TYPE_Rare': ["sum", "mean"],
        'CREDIT_TYPE_Consumer credit': ["sum", "mean"],
        'CREDIT_TYPE_Credit card': ["sum", "mean"],
        'NEW_STATUS_SCORE': ["sum"],
        'NEW_CRITICAL_STATUS': ["sum"]
    }

    # Applying aggregation operations to bureau _bb_agg table
    bureau_and_bb_agg = bureau_and_bb.groupby("SK_ID_CURR").agg(agg_list).reset_index()

    # Renaming the variables
    bureau_and_bb_agg.columns = pd.Index(
        ["BB_" + col[0] + "_" + col[1].upper() for col in bureau_and_bb_agg.columns.tolist()])

    # Expresses how many months on average they take out a loan
    bureau_and_bb_agg["BB_NEW_DAYS_CREDIT_RANGE"] = round(
        (bureau_and_bb_agg["BB_DAYS_CREDIT_MAX"] - bureau_and_bb_agg["BB_DAYS_CREDIT_MIN"]) / (
                30 * bureau_and_bb_agg["BB_SK_ID_CURR_COUNT"]))

    # total overdue amounts / total amount of loans
    bureau_and_bb_agg.loc[bureau_and_bb_agg["BB_AMT_CREDIT_SUM_SUM"] != 0, "BB_NEW_RATIO_AMTSUM_AMTOVERDUE"] = \
        bureau_and_bb_agg["BB_AMT_CREDIT_SUM_OVERDUE_SUM"] / bureau_and_bb_agg["BB_AMT_CREDIT_SUM_SUM"]

    # total remaining debt amounts / total loans amount
    bureau_and_bb_agg.loc[bureau_and_bb_agg["BB_AMT_CREDIT_SUM_SUM"] != 0, "BB_NEW_RATIO_AMTSUM_AMTDEBT"] = \
        bureau_and_bb_agg["BB_AMT_CREDIT_SUM_DEBT_SUM"] / bureau_and_bb_agg["BB_AMT_CREDIT_SUM_SUM"]

    # total overdue amounts / total outstanding debt amounts
    bureau_and_bb_agg.loc[bureau_and_bb_agg["BB_NEW_RATIO_AMTSUM_AMTDEBT"] != 0, "BB_NEW_RATIO_OVERDUE_AMTDEBT"] = \
        bureau_and_bb_agg["BB_AMT_CREDIT_SUM_OVERDUE_SUM"] / bureau_and_bb_agg["BB_AMT_CREDIT_SUM_DEBT_SUM"]

    # agg_list for the active and credits
    agg_list = {'DAYS_CREDIT': ['min', 'max', 'mean', 'var'],
                'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
                'DAYS_CREDIT_UPDATE': ['mean'],
                'CREDIT_DAY_OVERDUE': ['sum', 'max', 'mean'],
                'AMT_CREDIT_MAX_OVERDUE': ['mean'],
                'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
                'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
                'AMT_CREDIT_SUM_OVERDUE': ['mean'],
                'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
                'AMT_ANNUITY': ['max', 'mean'],
                'CNT_CREDIT_PROLONG': ['sum']}

    # Bureau: Active credits - using only numerical aggregations
    active = bureau_and_bb[bureau_and_bb['CREDIT_ACTIVE_Active'] == 1]
    active_agg = active.groupby('SK_ID_CURR').agg(agg_list)
    active_agg.columns = pd.Index(['BB_NEW_ACTIVE_' + e[0] + "_" + e[1].upper() for e in active_agg.columns.tolist()])
    bureau_and_bb_agg.rename(columns={'BB_SK_ID_CURR_': 'SK_ID_CURR'}, inplace=True)
    bureau_and_bb_agg = bureau_and_bb_agg.join(active_agg, how='left', on='SK_ID_CURR')

    # Bureau: Closed credits - using only numerical aggregations
    closed = bureau_and_bb[bureau_and_bb['CREDIT_ACTIVE_Closed'] == 1]
    closed_agg = closed.groupby('SK_ID_CURR').agg(agg_list)
    closed_agg.columns = pd.Index(['BB_NEW_CLOSED_' + e[0] + "_" + e[1].upper() for e in closed_agg.columns.tolist()])
    bureau_and_bb_agg = bureau_and_bb_agg.join(closed_agg, how='left', on='SK_ID_CURR')

    bureau_and_bb_agg.set_index('SK_ID_CURR', inplace=True)

    return bureau_and_bb_agg


def eda_credit_card():
    credit_card_balance = pd.read_csv("datasets/home-credit-default-risk/credit_card_balance.csv")

    def one_hot_encodery(df, nan_as_category=True):
        original_columns = list(df.columns)
        categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
        df = pd.get_dummies(df, columns=categorical_columns, dummy_na=nan_as_category)
        new_columns = [c for c in df.columns if c not in original_columns]
        return df, new_columns

    credit_card_balance, cat_cols = one_hot_encodery(credit_card_balance, nan_as_category=True)
    credit_card_balance.drop(['SK_ID_PREV'], axis=1, inplace=True)
    credit_card_balance["used_limit"] = credit_card_balance["AMT_BALANCE"] / credit_card_balance[
        "AMT_CREDIT_LIMIT_ACTUAL"] * 100
    # Ay boyunca limitin ne kadarını kullanmış.["used_limit"]

    # credit_card_balance['SK_ID_CURR'].value_counts(ascending=False, dropna=False)
    credit_card_balance["atm_ratio"] = credit_card_balance["AMT_DRAWINGS_ATM_CURRENT"] / credit_card_balance[
        "AMT_DRAWINGS_CURRENT"] * 100
    # Müşteri yaptığı çekimlerin ne kadarını ATM 'den yapmış
    # AMT_DRAWINGS_ATM_CURRENT--> kredi ayı boyunca ATM’den çekilen miktar
    # AMT_DRAWINGS_CURRENT	--> kredi ayı boyunca çekilen miktar

    credit_card_balance["other_ratio"] = credit_card_balance["AMT_DRAWINGS_OTHER_CURRENT"] / credit_card_balance[
        "AMT_DRAWINGS_CURRENT"] * 1000
    # çekimlerin ne kadarını diğer yöntemlerle yapmış

    credit_card_balance["pos_ratio"] = credit_card_balance["AMT_DRAWINGS_POS_CURRENT"] / credit_card_balance[
        "AMT_DRAWINGS_CURRENT"] * 100
    # çekimlerin ne kadarını POStan yapmış

    # last_prevs = credit_card_balance[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()

    # diğer kolonların ortalaması,min,max,sum'ı:

    credit_card_balance = credit_card_balance.groupby('SK_ID_CURR').agg(['min', 'max', 'mean', 'sum', 'var'])
    # credit_card_balance_ort.columns = ['cc_bal_' + i for i in credit_card_balance_ort.columns]
    credit_card_balance.columns = pd.Index(
        ['CC_BAL_' + col[0] + "_" + col[1].upper() for col in credit_card_balance.columns.tolist()])

    # Count credit card lines:Kredi kartı hatlarını sayısı
    credit_card_balance['CC_COUNT'] = credit_card_balance.groupby('SK_ID_CURR').size()

    # One hot encoding
    credit_card_balance = pd.get_dummies(credit_card_balance, dummy_na=True)
    # credit_card_balance['SK_ID_CURR'].value_counts(ascending=False)

    return credit_card_balance


def eda_pos_cash():
    pos = pd.read_csv('datasets/home-credit-default-risk/POS_CASH_balance.csv')

    def one_hot_encodery(df, nan_as_category=True):
        original_columns = list(df.columns)
        categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
        df = pd.get_dummies(df, columns=categorical_columns, dummy_na=nan_as_category)
        new_columns = [c for c in df.columns if c not in original_columns]
        return df, new_columns

    pos_balance, cat_cols = one_hot_encodery(pos, nan_as_category=True)
    pos_balance.drop(['SK_ID_PREV'], axis=1, inplace=True)
    pos_balance['cnt_ins_curr'] = pos_balance['CNT_INSTALMENT'] - pos_balance['CNT_INSTALMENT_FUTURE']

    pos_balance = pos_balance.groupby('SK_ID_CURR').agg(['min', 'max', 'mean', 'sum', 'var'])
    # credit_card_balance_ort.columns = ['cc_bal_' + i for i in credit_card_balance_ort.columns]
    pos_balance.columns = pd.Index(
        ['POS_' + col[0] + "_" + col[1].upper() for col in pos_balance.columns.tolist()])

    pos_balance.columns

    pos_balance['POS_NEW_IS_CREDIT_NOT_COMPLETED_ON_TIME'] = (pos_balance['POS_CNT_INSTALMENT_FUTURE_MIN'] == 0) & (
            pos_balance['POS_NAME_CONTRACT_STATUS_Completed_SUM'] == 0)

    # 1:kredi zamaninda kapanmamis 0:kredi zamaninda kapanmis

    pos_balance['POS_NEW_IS_CREDIT_NOT_COMPLETED_ON_TIME'] = pos_balance[
        'POS_NEW_IS_CREDIT_NOT_COMPLETED_ON_TIME'].astype(int)

    pos_balance.shape

    pos_balance['POS_COUNT'] = pos_balance.groupby('SK_ID_CURR').size()

    # One hot encoding
    pos_balance = pd.get_dummies(pos_balance, dummy_na=True)

    pos_balance.shape  # (337252, 82)

    pos_balance.head()

    return pos_balance


def eda_prev_app():
    def grab_col_names(dataframe, cat_th=10, car_th=20, ignore_vars=[]):
        # excluded columns
        exc_cols = []
        if type(ignore_vars) is not list:
            exc_cols.append(ignore_vars)
        else:
            exc_cols.extend(ignore_vars)
            # for i in ignore_vars:
            #    exc_cols.append(i)
        # print(exc_cols)

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

    # aşağıdaki iki fonksiyonda da quantilleri parametrik ayarladım
    def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
        quartile1 = dataframe[col_name].quantile(q1)
        quartile3 = dataframe[col_name].quantile(q3)
        interquantile_range = quartile3 - quartile1
        up_limit = quartile3 + 1.5 * interquantile_range
        low_limit = quartile1 - 1.5 * interquantile_range
        return low_limit, up_limit

    def replace_with_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
        low_limit, up_limit = outlier_thresholds(dataframe, col_name, q1, q3)
        dataframe.loc[(dataframe[col_name] < low_limit), col_name] = low_limit
        dataframe.loc[(dataframe[col_name] > up_limit), col_name] = up_limit

    # verisetini yükleme
    prev = pd.read_csv(r'datasets/home-credit-default-risk/previous_application.csv')

    cat_cols, num_cols, cat_but_car, exc_cols = grab_col_names(prev, ignore_vars=["SK_ID_CURR", "SK_ID_PREV", "index"])

    # 1. Outliers (Aykırı Değerler) Aykırıları 0.01 - 0.99 sınırlarıyla baskıladım
    for col in num_cols:
        replace_with_thresholds(prev, col, q1=0.01, q3=0.99)

    # XNA değeri içeren değişkenler var, bu değeri içeren tüm değişkenleri listeleyelim.
    cols_XNA = [col for col in cat_cols if prev.loc[prev[col] == "XNA", col].shape[0] > 0]

    # XNA içeren değişkenlerdeki XNA değer sayısı çok fazla application_train ve test'deki gibi drop edersek çok sayıda gözlem gidecek, en iyisi bunları bir sınıf olarak bırakmak.

    # benzer şekilde 365243 içeren nümerik değişkenler var
    cols_365243 = [col for col in num_cols if prev.loc[prev[col] == 365243, col].shape[0] > 0]

    # ['DAYS_FIRST_DRAWING',
    #  'DAYS_FIRST_DUE',
    #  'DAYS_LAST_DUE_1ST_VERSION',
    #  'DAYS_LAST_DUE',
    #  'DAYS_TERMINATION']

    # bunları NaN ile değiştirelim
    for col in cols_365243:
        prev[col].replace(365243, np.nan, inplace=True)

    # 2. Missing Values (Eksik Değerler) Eksik değerlere müdahale etmedim

    # 3. Encoding (Binary-Label Encoding, One-Hot Encoding, Rare Encoding)

    binary_cols = [col for col in prev.columns if prev[col].dtype not in [int, float]
                   and len(prev[col].unique()) == 2]

    for col in binary_cols:
        label_encoder(prev, col)

    # tekrar değişken sınıflarını elde edelim

    cat_cols, num_cols, cat_but_car, exc_cols = grab_col_names(prev,
                                                               ignore_vars=["SK_ID_CURR", "SK_ID_PREV", "index"])

    # cat_but_car değişkenleri sınıf sayısı çok olan iki değişken, bunları kategoriklere ekleyip one-hot-encoding yapabiliriz
    # sütun sayısını arttırıyor ama faydasız ve rare değişkenleri elediğimizde azalacak (RARE ENCODING YAPMADIM HENÜZ)
    cat_cols = cat_cols + cat_but_car

    prev = one_hot_encoder_c(prev, cat_cols, drop_first=True, nan_as_category=True)

    cat_cols, num_cols, cat_but_car, exc_cols = grab_col_names(prev,
                                                               ignore_vars=["SK_ID_CURR", "SK_ID_PREV", "index"])

    # iki sınıflı ve sınıflardan birinin oranı % 1'den düşük olan değişkenleri faydasız olarak belirleyelim
    useless_cols = [col for col in prev.columns if len(prev[col].unique()) == 2 and
                    (prev[col].value_counts() / len(prev) < 0.01).any(axis=None)]

    # bunları silmeden önce verisetini yedekleyelim
    prev1 = prev.copy()

    # bu değişkenleri düşürelim 60 değişken düşecek
    prev1.drop(useless_cols, inplace=True, axis=1)

    cat_cols, num_cols, cat_but_car, exc_cols = grab_col_names(prev1,
                                                               ignore_vars=["SK_ID_CURR", "SK_ID_PREV", "index"])

    # bu sefer tek sınıf kalmış olanları silelim
    useless_cols_2 = [col for col in prev1.columns if len(prev1[col].unique()) == 1]

    # bu şekilde de 15 değişken düşecek
    prev1.drop(useless_cols_2, axis=1, inplace=True)

    cat_cols, num_cols, cat_but_car, exc_cols = grab_col_names(prev1,
                                                               ignore_vars=["SK_ID_CURR", "SK_ID_PREV", "index"])
    # bu değişikliklerin ardından 91 değişken kalıyor 72'si kategorik 17'si nümerik

    prev2 = prev1.copy()

    # 4. Feature Scaling (Özellik Ölçeklendirme) BUNU ŞİMDİLİK UYGULAMIYORUM

    # 5. Feature Extraction (Özellik Çıkarımı)

    # Müşterinin önceki başvuruda istediği kredi miktarı / aldığı nihai kredi miktarı
    prev2['NEW_APP_CREDIT_PERC'] = prev2['AMT_APPLICATION'] / prev2['AMT_CREDIT']
    # Müşterinin önceki başvuruda istediği kredi miktarı - aldığı nihai kredi miktarı
    prev2['NEW_APPLICATION_CREDIT_DIFF'] = prev2['AMT_APPLICATION'] - prev2['AMT_CREDIT']
    # Müşterinin önceki başvuruda aldığı nihai kredi miktarı / bankaya gönderdiği aylık ödeme
    prev2['NEW_CREDIT_TO_ANNUITY_RATIO'] = prev2['AMT_CREDIT'] / prev2['AMT_ANNUITY']
    # Müşterinin başvurduğu kredi harici ödediği peşinat miktarı / aldığı nihai kredi miktarı
    prev2['NEW_DOWN_PAYMENT_TO_CREDIT'] = prev2['AMT_DOWN_PAYMENT'] / prev2['AMT_CREDIT']
    # Müşterinin önceki başvuruda aldığı nihai kredi miktarı / Tüketici kredisinin verildiği malın fiyatı
    prev2["NEW_CREDIT_TO_GOODS_RATIO"] = prev2["AMT_CREDIT"] / prev2["AMT_GOODS_PRICE"]

    cat_cols, num_cols, cat_but_car, exc_cols = grab_col_names(prev2,
                                                               ignore_vars=["SK_ID_CURR", "SK_ID_PREV", "index"])

    # aşağıdaki aggregation fonksiyonları üzerinde çok durmadım, ekleme çıkarma yapılabilir,
    # en son feature importance'a göre eleme yaptığım için denemelerimde gereksiz olanları atarım diye düşündüm
    # denedikçe gelişme olduğunda paylaşacağım

    num_aggregations = {
        # Önceki başvuruda (ÖB) bankaya gönderdiği aylık ödeme
        'AMT_ANNUITY': ['min', 'max', 'mean', 'sum'],
        # ÖB istediği kredi miktarı
        'AMT_APPLICATION': ['min', 'max', 'mean', 'sum'],
        # ÖB aldığı nihai kredi miktarı
        'AMT_CREDIT': ['min', 'max', 'mean', 'sum'],
        # ÖB kredi harici ödediği peşinat miktarı
        'AMT_DOWN_PAYMENT': ['min', 'max', 'mean', 'sum'],
        # ÖB tüketici kredisinin verildiği malın fiyatı
        'AMT_GOODS_PRICE': ['min', 'max', 'mean', 'sum'],
        # ÖB başvuru yaptığı saat
        'HOUR_APPR_PROCESS_START': ['min', 'max', 'mean'],
        # ÖB kredi harici ödediği peşinat miktarının oranı
        'RATE_DOWN_PAYMENT': ['min', 'max', 'mean'],
        # ÖB bir önceki başvuruda karar ne zaman alındı
        'DAYS_DECISION': ['min', 'max', 'mean'],
        # ÖB bir önceki başvurunun süresi
        'CNT_PAYMENT': ['mean', 'sum'],
        # yeni özellik
        'NEW_APP_CREDIT_PERC': ['min', 'max', 'mean', 'var'],
        # yeni özellik
        'NEW_APPLICATION_CREDIT_DIFF': ['min', 'max', 'mean', 'sum'],
        # yeni özellik
        'NEW_CREDIT_TO_ANNUITY_RATIO': ['mean', 'max'],
        # yeni özellik
        'NEW_DOWN_PAYMENT_TO_CREDIT': ['mean'],
        # yeni özellik
        'NEW_CREDIT_TO_GOODS_RATIO': ['mean', 'max'],
        # Önceki başvuruların adedi (eşsiz SK_ID_PREV sayarak)
        'SK_ID_PREV': ['nunique'],
        # ÖB bir önceki başvurunun beklenen sonlandırılması ne zaman oldu
        'DAYS_TERMINATION': ['max']
    }

    cat_aggregations = {}
    for cat in cat_cols:
        cat_aggregations[cat] = ['mean']

    prev_agg = prev2.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    prev_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])

    return prev_agg


def eda_installments_payments(num_rows=None):
    # print("Installments Payment Table Preprocessing is started")
    ins = pd.read_csv("datasets/home-credit-default-risk/installments_payments.csv", nrows=num_rows)
    ins, cat_cols = one_hot_encoder_z(ins, nan_as_category=True)
    # Percentage and difference paid in each installment (amount paid and installment value)(Her kredi taksidi ödemesinde ödediği miktarla aslı arasındaki fark ve bunun yüzdesi)
    ins['NEW_PAYMENT_PERC'] = ins['AMT_PAYMENT'] / ins['AMT_INSTALMENT']
    ins['NEW_PAYMENT_DIFF'] = ins['AMT_INSTALMENT'] - ins['AMT_PAYMENT']
    # Days past due and days before due (no negative values)
    ins['NEW_DPD'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']
    ins['NEW_DBD'] = ins['DAYS_INSTALMENT'] - ins['DAYS_ENTRY_PAYMENT']
    ins['NEW_DPD'] = ins['NEW_DPD'].apply(lambda x: x if x > 0 else 0)
    ins['NEW_DBD'] = ins['NEW_DBD'].apply(lambda x: x if x > 0 else 0)
    # Features: Perform aggregations
    aggregations = {
        'NUM_INSTALMENT_VERSION': ['nunique'],
        'NEW_DPD': ['max', 'mean', 'sum'],
        'NEW_DBD': ['max', 'mean', 'sum'],
        'NEW_PAYMENT_PERC': ['max', 'mean', 'sum', 'var'],
        'NEW_PAYMENT_DIFF': ['max', 'mean', 'sum', 'var'],
        'AMT_INSTALMENT': ['max', 'mean', 'sum'],
        'AMT_PAYMENT': ['min', 'max', 'mean', 'sum'],
        'DAYS_ENTRY_PAYMENT': ['max', 'mean', 'sum']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']
    ins_agg = ins.groupby('SK_ID_CURR').agg(aggregations)
    ins_agg.columns = pd.Index(['NEW_INSTAL_' + e[0] + "_" + e[1].upper() for e in ins_agg.columns.tolist()])

    # Count installments accounts
    ins_agg['NEW_INSTAL_COUNT'] = ins.groupby('SK_ID_CURR').size()
    del ins
    gc.collect()
    # print("Installments Payment Table Preprocessing is finished")
    return ins_agg


# -- Model Base
def combine_files():
    bureau_and_bb_agg = eda_bureau_bb()

    pos_agg = eda_pos_cash()

    credit_card_b_df = eda_credit_card()

    prev_agg_df = eda_prev_app()

    installment_df = eda_installments_payments()

    application = eda_application()

    print("application shape: ", application.shape)
    print("bureau_and_bb_agg shape: ", bureau_and_bb_agg.shape)
    print("pos_agg shape: ", pos_agg.shape)
    print("credit_card_b_df shape: ", credit_card_b_df.shape)
    print("prev_agg_df shape: ", prev_agg_df.shape)
    print("installment_df shape: ", installment_df.shape)

    df = application.join(bureau_and_bb_agg, how='left', on='SK_ID_CURR')
    df = df.join(pos_agg, how='left', on='SK_ID_CURR')
    df = df.join(credit_card_b_df, how='left', on='SK_ID_CURR')
    df = df.join(prev_agg_df, how='left', on='SK_ID_CURR')
    df = df.join(installment_df, how='left', on='SK_ID_CURR')
    print("df_final: ", df.shape)
    del application, bureau_and_bb_agg, pos_agg, credit_card_b_df, prev_agg_df, installment_df
    gc.collect()

    df = df.replace([-np.inf, np.inf], np.nan)
    col_sca = [col for col in df.columns if col not in ['TARGET', 'SK_ID_CURR'] and df[col].nunique() > 2]
    scaler = MinMaxScaler()
    df[col_sca] = scaler.fit_transform(df[col_sca])

    # df_final[df_final.duplicated()]
    # Empty DataFrame

    return df


def run_model_base(df):
    ######################################
    # 9. Modeling
    ######################################
    global train_df, test_df
    df = df.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
    train_df = df[df['TARGET'].notnull()]
    test_df = df[df['TARGET'].isnull()]

    global X, y, X_train, X_test, y_train, y_test
    y = train_df["TARGET"]
    X = train_df.drop(["SK_ID_CURR", "TARGET"], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)
    model = LGBMClassifier(random_state=1).fit(X_train, y_train)

    imp = plot_importance(model, X, 10)

    y_prob = model.predict_proba(X_test)[:, 1]
    print("\n")
    print("roc_auc_score: ", round(roc_auc_score(y_test, y_prob), 4))

    sub_x = test_df.drop(["SK_ID_CURR", "TARGET"], axis=1)
    test_df['TARGET'] = model.predict_proba(sub_x)[:, 1]
    test_df[['SK_ID_CURR', 'TARGET']].to_csv('submission.csv', index=False)

    return test_df


def run_model_base_CAT(df):
    ######################################
    # 9. Modeling
    ######################################
    global train_df, test_df
    df = df.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
    train_df = df[df['TARGET'].notnull()]
    test_df = df[df['TARGET'].isnull()]
    global X, y, X_train, X_test, y_train, y_test
    y = train_df["TARGET"]
    X = train_df.drop(["SK_ID_CURR", "TARGET"], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)
    t0 = time.time()
    model = CatBoostClassifier(random_state=1).fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]
    print("roc_auc_score: ", round(roc_auc_score(y_test, y_prob), 4))
    time_taken_minutes = (time.time() - t0) / 60
    print(f'Total time taken in training: ', round(time_taken_minutes, 2), 'minutes!')
    return test_df


def run_model_search(df):
    ######################################################
    global train_df, test_df
    df = df.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
    train_df = df[df['TARGET'].notnull()]
    test_df = df[df['TARGET'].isnull()]

    train_df = train_df.fillna(-1)
    filcols = [col for col in test_df.columns if 'TARGET' not in col]
    test_df[filcols] = test_df[filcols].fillna(-1)

    global X, y, X_train, X_test, y_train, y_test
    y = train_df["TARGET"]
    X = train_df.drop(["SK_ID_CURR", "TARGET"], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)
    ###############################################################

    models = [
        ('LogiReg', LogisticRegression()),
        # ('KNN', KNeighborsClassifier()), # taking so long
        ('CART', DecisionTreeClassifier()),
        ('RF', RandomForestClassifier()),
        # ('SVR', SVC()), # taking so long
        ('GBM', GradientBoostingClassifier()),
        ("XGBoost", XGBClassifier()),
        ("LightGBM", LGBMClassifier()),
        ("CatBoost", CatBoostClassifier(verbose=False)),
        ("AdaBoost", AdaBoostClassifier()),
        ("Bagging", BaggingClassifier()),
        ("ExtraTrees", ExtraTreesClassifier()),
        ("HistGradient", HistGradientBoostingClassifier())
    ]
    # regressor = CatBoostClassifier(verbose=False)
    global output_df
    output_df = pd.DataFrame(models, columns=["MODEL_NAME", "MODEL_BASE"])
    output_df.drop('MODEL_BASE', axis=1, inplace=True)
    y = y.astype(float)
    for name, regressor in models:
        t0 = time.time()
        print("Running Base--> ", name)
        roc_auc = np.mean(cross_val_score(regressor, X, y, cv=2, scoring="roc_auc"))
        time_taken_minutes = (time.time() - t0) / 60
        print(f'Total time taken in {name} training: ', time_taken_minutes, 'minutes!')
        print(f"roc_auc: {round(roc_auc, 4)} ({name}) ")
        output_df.loc[output_df['MODEL_NAME'] == name, "roc_auc_CV_Base"] = roc_auc
        output_df.loc[output_df['MODEL_NAME'] == name, "time_minutes_base"] = time_taken_minutes

    print(output_df)
    return output_df


def run_model_selected_cols(dfin, col_num=10):
    ######################################
    # 9. Modeling
    ######################################
    df = dfin.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
    train_df = df[df['TARGET'].notnull()]
    test_df = df[df['TARGET'].isnull()].drop("TARGET", axis=1)
    y = train_df["TARGET"]
    X = train_df.drop(["SK_ID_CURR", "TARGET"], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)
    model = LGBMClassifier(random_state=1).fit(X_train, y_train)

    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': X.columns})
    feature_imp = feature_imp.sort_values(by="Value", ascending=False).reset_index(drop=True)
    feature_imp = feature_imp[0:col_num]
    imp = feature_imp['Feature'].values.tolist()
    imp.append('TARGET')
    df = df[imp]

    train_df = df[df['TARGET'].notnull()]
    test_df = df[df['TARGET'].isnull()].drop("TARGET", axis=1)
    y = train_df["TARGET"]
    X = train_df.drop(["TARGET"], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)
    model = LGBMClassifier(random_state=1).fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    print("\n")
    print(f"roc_auc_score with {col_num} columns: ", round(roc_auc_score(y_test, y_prob), 4))


# -- Model Tunning
def param_search_all(df):
    global train_df, test_df
    df = df.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
    train_df = df[df['TARGET'].notnull()]
    test_df = df[df['TARGET'].isnull()]

    # train_df = train_df.fillna(-1)
    # filcols = [col for col in test_df.columns if 'TARGET' not in col]
    # test_df[filcols] = test_df[filcols].fillna(-1)

    global X, y, X_train, X_test, y_train, y_test
    y = train_df["TARGET"]
    X = train_df.drop(["SK_ID_CURR", "TARGET"], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

    models = [  # ('LogiReg', LogisticRegression()),
        # ('KNN', KNeighborsClassifier()),
        # ('CART', DecisionTreeClassifier()),
        # ('RF', RandomForestClassifier()),
        # ('SVR', SVC()),
        # ('GBM', GradientBoostingClassifier()),
        # ("XGBoost", XGBClassifier(objective='reg:squarederror')),
        ("LightGBM", LGBMClassifier()),
        ("CatBoost", CatBoostClassifier(verbose=False)),
        # ("AdaBoost", AdaBoostClassifier()),
        # ("Bagging", BaggingClassifier()),
        # ("ExtraTrees", ExtraTreesClassifier()),
        ("HistGradient", HistGradientBoostingClassifier()),
    ]

    global output_df
    output_df = pd.DataFrame(models, columns=["MODEL_NAME", "MODEL_BASE"])
    output_df.drop('MODEL_BASE', axis=1, inplace=True)
    for name, regressor in models:
        t0 = time.time()
        print("Running Base--> ", name)
        rmse_cv = np.mean(cross_val_score(regressor, X, y, cv=3, scoring="roc_auc"))
        time_taken_minutes = (time.time() - t0) / 60
        print(f'Total time taken in {name} training: ', time_taken_minutes, 'minutes!')
        print(f"roc_auc: {round(rmse_cv, 4)} ({name}) ")
        output_df.loc[output_df['MODEL_NAME'] == name, "roc_auc_CV_Base"] = rmse_cv
        output_df.loc[output_df['MODEL_NAME'] == name, "time_minutes_base"] = time_taken_minutes

    # HYPER PARAMETERS TUNNING
    rf_params = {"max_depth": [5, 15, None],
                 "max_features": [5, 9, "auto"],
                 "min_samples_split": [6, 8, 15],
                 "n_estimators": [150, 200, 300]}

    xgboost_params = {"learning_rate": [0.01, 0.1, 0.15],
                      "max_depth": [3, 5, 8],
                      "n_estimators": [100, 200, 300],
                      "colsample_bytree": [0.3, 0.5, 0.8]}

    lightgbm_params = {"learning_rate": [0.001, 0.01, 0.1],
                       "n_estimators": [100, 300, 500],
                       "colsample_bytree": [0.1, 0.3, 0.7, 1],
                       # "num_leaves": [5,31],
                       # "max_bin": [55, 255],
                       # "bagging_fraction": [0.8, 1.0],
                       # "bagging_freq": [0, 5],
                       # "feature_fraction": [0.2319, 1.0],
                       # "feature_fraction_seed": [2, 9],
                       # "bagging_seed": [3, 9],
                       # "min_data_in_leaf": [6, 20],
                       # "min_sum_hessian_in_leaf": [0.003, 11]
                       }

    extraTrees_params = {
        'n_estimators': [10, 50, 100],
        'max_depth': [2, 16, 50],
        'min_samples_split': [2, 6],
        'min_samples_leaf': [1, 2],
        'max_features': ['auto', 'sqrt', 'log2'],
        'bootstrap': [True, False],
        'warm_start': [True, False],
    }

    HistGradient_params = {
        "learning_rate": [0.01, 0.05],
        "max_iter": [20, 100],
        "max_depth": [None, 25],
        "l2_regularization": [0.0, 1.5],
    }

    catboost_params = {
        "iterations": [200, 500, 1000],
        "learning_rate": [0.01, 0.1, 0.3],
        "depth": [3, 6]
    }

    regressors = [
        # ("RF", RandomForestClassifier(), rf_params),
        # ('XGBoost', XGBClassifier(objective='binary:logistic'), xgboost_params),
        ('LightGBM', LGBMClassifier(), lightgbm_params),
        ('CatBoost', CatBoostClassifier(verbose=False), catboost_params),
        # ('ExtraTrees', ExtraTreesClassifier(), extraTrees_params),
        ('HistGradient', HistGradientBoostingClassifier(), HistGradient_params),
    ]
    global best_models
    best_models = {}

    for name, regressor, params in regressors:
        t0 = time.time()
        print("Running ParamSearch --> ", name)
        # GridSearch
        # gs_best = GridSearchCV(regressor, params, cv=3, n_jobs=-1, verbose=False).fit(X, y)
        gs_best = RandomizedSearchCV(regressor, params, cv=3, n_jobs=-1, verbose=False).fit(X, y)

        final_model = regressor.set_params(**gs_best.best_params_)
        rmse = np.mean((cross_val_score(final_model, X, y, cv=3, scoring="roc_auc")))
        time_taken_minutes = (time.time() - t0) / 60
        print(f'Total time taken in {name} training: ', round(time_taken_minutes, 2), 'minutes!')
        print(f"roc_auc (After): {round(rmse, 4)} ({name}) ")
        print(f"{name} best params: {gs_best.best_params_}", end="\n\n")
        output_df.loc[output_df['MODEL_NAME'] == name, "roc_auc_CV_PARAM"] = rmse
        output_df.loc[output_df['MODEL_NAME'] == name, "BEST_PARAMS"] = str(gs_best.best_params_)
        output_df.loc[output_df['MODEL_NAME'] == name, "time_minutes_tunning"] = time_taken_minutes

        best_models[name] = final_model

    print(output_df)
    print(best_models)
    # sub_x = test_df.drop(["SK_ID_CURR", "TARGET"], axis=1)
    # test_df['TARGET'] = final_model.predict_proba(sub_x)[:, 1]
    # test_df[['SK_ID_CURR', 'TARGET']].to_csv('submission.csv', index=False)

    return output_df, best_models


def param_search_lgbm(df):
    # df = df_final.copy()
    global train_df, test_df
    df = df.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
    train_df = df[df['TARGET'].notnull()]
    test_df = df[df['TARGET'].isnull()]

    # train_df = train_df.fillna(-1)
    # filcols = [col for col in test_df.columns if 'TARGET' not in col]
    # test_df[filcols] = test_df[filcols].fillna(-1)

    global X, y, X_train, X_test, y_train, y_test
    y = train_df["TARGET"]
    X = train_df.drop(["SK_ID_CURR", "TARGET"], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

    models = [  # ('LogiReg', LogisticRegression()),
        # ('KNN', KNeighborsClassifier()),
        # ('CART', DecisionTreeClassifier()),
        # ('RF', RandomForestClassifier()),
        # ('SVR', SVC()),
        # ('GBM', GradientBoostingClassifier()),
        # ("XGBoost", XGBClassifier(objective='reg:squarederror')),
        ("LightGBM", LGBMClassifier(random_state=1)),
        # ("CatBoost", CatBoostClassifier(verbose=False)),
        # ("AdaBoost", AdaBoostClassifier()),
        # ("Bagging", BaggingClassifier()),
        # ("ExtraTrees", ExtraTreesClassifier()),
        # ("HistGradient", HistGradientBoostingClassifier()),
    ]

    global output_df
    output_df = pd.DataFrame(models, columns=["MODEL_NAME", "MODEL_BASE"])
    output_df.drop('MODEL_BASE', axis=1, inplace=True)
    # for name, regressor in models:
    #     t0 = time.time()
    #     print("Running Base--> ", name)
    #     rmse_cv = np.mean(cross_val_score(regressor, X, y, cv=3, scoring="roc_auc"))
    #     time_taken_minutes = (time.time() - t0) / 60
    #     print(f'Total time taken in {name} training: ', time_taken_minutes, 'minutes!')
    #     print(f"roc_auc: {round(rmse_cv, 4)} ({name}) ")
    #     output_df.loc[output_df['MODEL_NAME'] == name, "roc_auc_CV_Base"] = rmse_cv
    #     output_df.loc[output_df['MODEL_NAME'] == name, "time_minutes_base"] = time_taken_minutes

    # HYPER PARAMETERS TUNNING
    # rf_params = {"max_depth": [5, 15, None],
    #              "max_features": [5, 9, "auto"],
    #              "min_samples_split": [6, 8, 15],
    #              "n_estimators": [150, 200, 300]}
    #
    # xgboost_params = {"learning_rate": [0.01, 0.1, 0.15],
    #                   "max_depth": [3, 5, 8],
    #                   "n_estimators": [100, 200, 300],
    #                   "colsample_bytree": [0.3, 0.5, 0.8]}

    # roc_auc(After): 0.7830(LightGBM) {'n_estimators': 300, 'learning_rate': 0.1, 'colsample_bytree': 0.1}
    # roc_auc(After): 0.7719(LightGBM) {'n_estimators': 300, 'learning_rate': 0.2, 'colsample_bytree': 0.1}
    # roc_auc (After): 0.7867 (LightGBM) {'colsample_bytree': 0.7, 'max_depth': 3, 'n_estimators': 500}
    #                    "learning_rate": [0.001, 0.01, 0.1],
    #                    "n_estimators": [100, 300, 500],
    #                    "colsample_bytree": [0.1, 0.3, 0.7, 1],

    lightgbm_params = {
        "learning_rate": [0.1],  # 0.1
        "n_estimators": [1000],
        'max_depth': [8],
        "colsample_bytree": [0.3],
        # "num_leaves": [5,31],
        # "max_bin": [55, 255],
        # "bagging_fraction": [0.8, 1.0],
        # "bagging_freq": [0, 5],
        # "feature_fraction": [0.2319, 1.0],
        # "feature_fraction_seed": [2, 9],
        # "bagging_seed": [3, 9],
        # "min_data_in_leaf": [6, 20],
        # "min_sum_hessian_in_leaf": [0.003, 11]
    }

    # lightgbm_params = {
    #                    "learning_rate": [0.001, 0.01, 0.1],
    #                    "n_estimators": [100, 300, 500],
    #                    "colsample_bytree": [0.1, 0.3, 0.7, 1],
    #                    # "num_leaves": [5,31],
    #                    # "max_bin": [55, 255],
    #                    # "bagging_fraction": [0.8, 1.0],
    #                    # "bagging_freq": [0, 5],
    #                    # "feature_fraction": [0.2319, 1.0],
    #                    # "feature_fraction_seed": [2, 9],
    #                    # "bagging_seed": [3, 9],
    #                    # "min_data_in_leaf": [6, 20],
    #                    # "min_sum_hessian_in_leaf": [0.003, 11]
    #                    }

    # extraTrees_params = {
    #     'n_estimators': [10, 50, 100],
    #     'max_depth': [2, 16, 50],
    #     'min_samples_split': [2, 6],
    #     'min_samples_leaf': [1, 2],
    #     'max_features': ['auto', 'sqrt', 'log2'],
    #     'bootstrap': [True, False],
    #     'warm_start': [True, False],
    # }

    HistGradient_params = {
        "learning_rate": [0.01, 0.05],
        "max_iter": [20, 100],
        "max_depth": [None, 25],
        "l2_regularization": [0.0, 1.5],
    }

    catboost_params = {
        "iterations": [200, 500, 1000],
        "learning_rate": [0.01, 0.1, 0.3],
        "depth": [3, 6]
    }

    regressors = [
        # ("RF", RandomForestClassifier(), rf_params),
        # ('XGBoost', XGBClassifier(objective='binary:logistic'), xgboost_params),
        ('LightGBM', LGBMClassifier(random_state=1), lightgbm_params),
        # ('CatBoost', CatBoostClassifier(verbose=False), catboost_params),
        # ('ExtraTrees', ExtraTreesClassifier(), extraTrees_params),
        # ('HistGradient', HistGradientBoostingClassifier(), HistGradient_params),
    ]
    global best_models
    best_models = {}

    for name, regressor, params in regressors:
        t0 = time.time()
        print("Running ParamSearch --> ", name)
        # GridSearch
        # gs_best = GridSearchCV(regressor, params, cv=3, n_jobs=-1, verbose=False).fit(X, y)

        # gs_best = RandomizedSearchCV(regressor, params, cv=3, n_jobs=-1, verbose=False).fit(X, y)

        # final_model = regressor.set_params(**gs_best.best_params_)
        final_model = LGBMClassifier(random_state=1, colsample_bytree=0.3, n_estimators=1000, learning_rate=0.1,
                                     max_depth=8)

        rmse = np.mean((cross_val_score(final_model, X, y, cv=3, scoring="roc_auc")))
        # cv_results = cross_validate(final_model, X, y, cv=3, scoring=["accuracy", "f1", "roc_auc", "recall", "precision"])

        time_taken_minutes = (time.time() - t0) / 60
        print(f'Total time taken in {name} training: ', round(time_taken_minutes, 2), 'minutes!')
        print(f"roc_auc (After): {round(rmse, 4)} ({name}) ")

        # print('test_accuracy', cv_results['test_accuracy'].mean())
        # print('test_f1', cv_results['test_f1'].mean())
        # print('test_roc_auc', cv_results['test_roc_auc'].mean())
        # print('test_recall', cv_results['test_recall'].mean())
        # print('test_precision', cv_results['test_precision'].mean())

        # print(f"{name} best params: {gs_best.best_params_}", end="\n\n")
        print(f"{name} best params: {params}", end="\n\n")
        output_df.loc[output_df['MODEL_NAME'] == name, "roc_auc_CV_PARAM"] = rmse
        # output_df.loc[output_df['MODEL_NAME'] == name, "roc_auc_CV_PARAM"] = cv_results['test_roc_auc'].mean()
        # output_df.loc[output_df['MODEL_NAME'] == name, "BEST_PARAMS"] = str(gs_best.best_params_)
        output_df.loc[output_df['MODEL_NAME'] == name, "BEST_PARAMS"] = str(params)
        output_df.loc[output_df['MODEL_NAME'] == name, "time_minutes_tunning"] = time_taken_minutes

        best_models[name] = final_model

    print(output_df)
    print(best_models)
    # sub_x = test_df.drop(["SK_ID_CURR", "TARGET"], axis=1)
    # test_df['TARGET'] = final_model.predict_proba(sub_x)[:, 1]
    # test_df[['SK_ID_CURR', 'TARGET']].to_csv('submission.csv', index=False)

    return output_df, best_models


def param_search_lgbm_earlystop(df):
    df = df.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
    train_df = df[df['TARGET'].notnull()]
    test_df = df[df['TARGET'].isnull()]

    train_df = train_df.fillna(-1)
    filcols = [col for col in test_df.columns if 'TARGET' not in col]
    test_df[filcols] = test_df[filcols].fillna(-1)

    y = train_df["TARGET"]
    X = train_df.drop(["SK_ID_CURR", "TARGET"], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)
    model = LGBMClassifier(nthread=-1,
                           # device_type='gpu',
                           n_estimators=5000,
                           learning_rate=0.01,
                           max_depth=11,
                           num_leaves=58,
                           colsample_bytree=0.613,
                           subsample=0.708,
                           max_bin=407,
                           reg_alpha=3.564,
                           reg_lambda=4.930,
                           min_child_weight=6,
                           min_child_samples=165,
                           # keep_training_booster=True,
                           silent=-1,
                           verbose=-1)
    lab_enc = LabelEncoder()
    y_train = lab_enc.fit_transform(y_train)
    # train_predict(clf, samples, X_train, X_test, y_train, y_test)
    # , first_metric_only = True
    # model.fit(X_train, X_test, eval_set=[(X_train, y_train), (X_test, y_test)], eval_metric='auc', verbose=100,
    #           early_stopping_rounds=300)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric=['auc'], verbose=100,
              early_stopping_rounds=300)

    sub_x = test_df.drop(["SK_ID_CURR", "TARGET"], axis=1)
    test_df['TARGET'] = model.predict_proba(sub_x)[:, 1]
    test_df[['SK_ID_CURR', 'TARGET']].to_csv('submission.csv', index=False)


def param_search_catboost(df):
    # df = df_final.copy()
    global train_df, test_df
    df = df.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
    train_df = df[df['TARGET'].notnull()]
    test_df = df[df['TARGET'].isnull()]

    # train_df = train_df.fillna(-1)
    # filcols = [col for col in test_df.columns if 'TARGET' not in col]
    # test_df[filcols] = test_df[filcols].fillna(-1)

    y = train_df["TARGET"]
    X = train_df.drop(["SK_ID_CURR", "TARGET"], axis=1)

    models = [("CatBoost", CatBoostClassifier(verbose=False))]

    global output_df
    output_df = pd.DataFrame(models, columns=["MODEL_NAME", "MODEL_BASE"])
    output_df.drop('MODEL_BASE', axis=1, inplace=True)

    # {"iterations": [400, 600],"learning_rate": [0.08, 0.15],"depth": [5, 8],'loss_function': ['Logloss', 'CrossEntropy']
    # params: {'loss_function': 'Logloss', 'learning_rate': 0.08, 'iterations': 600, 'depth': 8}
    # roc_auc(After): 0.7863(CatBoost) kaggle 0.79076
    # --
    #
    #

    catboost_params = {
        "iterations": [800, 1000],
        "learning_rate": [0.04, 0.06],
        "depth": [10, 12],
        'loss_function': ['Logloss']
    }
    # {'loss_function': 'Logloss', 'learning_rate': 0.04, 'iterations': 800, 'depth': 10}

    regressors = [('CatBoost', CatBoostClassifier(verbose=False), catboost_params)]
    global best_models
    best_models = {}

    for name, regressor, params in regressors:
        t0 = time.time()
        print("Running ParamSearch --> ", name)
        # gs_best = GridSearchCV(regressor, params, cv=3, n_jobs=-1, verbose=False).fit(X, y)
        gs_best = RandomizedSearchCV(regressor, params, cv=2, n_jobs=-1, verbose=False).fit(X, y)
        final_model = regressor.set_params(**gs_best.best_params_)
        rmse = np.mean((cross_val_score(final_model, X, y, cv=2, scoring="roc_auc")))

        time_taken_minutes = (time.time() - t0) / 60
        print(f'Total time taken in {name} training: ', round(time_taken_minutes, 2), 'minutes!')
        print(f"roc_auc (After): {round(rmse, 4)} ({name}) ")
        print(f"{name} best params: {gs_best.best_params_}", end="\n\n")

        output_df.loc[output_df['MODEL_NAME'] == name, "roc_auc_CV_PARAM"] = rmse
        output_df.loc[output_df['MODEL_NAME'] == name, "BEST_PARAMS"] = str(gs_best.best_params_)
        output_df.loc[output_df['MODEL_NAME'] == name, "time_minutes_tunning"] = time_taken_minutes
        best_models[name] = final_model

    print(output_df)
    print(best_models)
    sub_x = test_df.drop(["SK_ID_CURR", "TARGET"], axis=1)
    test_df['TARGET'] = final_model.predict_proba(sub_x)[:, 1]
    test_df[['SK_ID_CURR', 'TARGET']].to_csv('submission.csv', index=False)

    return output_df, best_models


def model_stacking_all():
    ######################################################
    # # Stacking & Ensemble Learning
    ######################################################
    best_models
    global output_df

    classifiers = [
        ("voting_LGBM_Cat",
         VotingClassifier(estimators=[('LightGBM', best_models["LightGBM"]), ('CatBoost', best_models["CatBoost"])])),

        ('voting_LGBM_HIST', VotingClassifier(
            estimators=[('LightGBM', best_models["LightGBM"]), ('HistGradient', best_models["HistGradient"])])),

        ('voting_Cat_HIST', VotingClassifier(
            estimators=[('CatBoost', best_models["CatBoost"]), ('HistGradient', best_models["HistGradient"])])),

        ('voting_LGBM_CAT_HIST',
         VotingClassifier(estimators=[('LightGBM', best_models["LightGBM"]), ('CatBoost', best_models["CatBoost"])
             , ('HistGradient', best_models["HistGradient"])])),

        ('stacking_Cat_HIST_LGBM', StackingClassifier(
            estimators=[('CatBoost', best_models["CatBoost"]), ('HistGradient', best_models["HistGradient"])]
            , final_estimator=best_models["LightGBM"])),

        ('stacking_LGBM_HIST_Cat', StackingClassifier(
            estimators=[('LightGBM', best_models["LightGBM"]), ('HistGradient', best_models["HistGradient"])]
            , final_estimator=best_models["CatBoost"]))]

    for name, regressor in classifiers:
        t0 = time.time()
        print("Running Voting/Stacking --> ", name)
        model = regressor
        model.fit(X, y)
        rmse = np.mean((cross_val_score(model, X, y, cv=3, scoring="roc_auc")))
        print(f"roc_auc ({name}): {round(rmse, 4)} ")
        time_taken_minutes = (time.time() - t0) / 60
        print(f'Total time taken in {name} training: ', round(time_taken_minutes, 2), 'minutes!')
        output_df = output_df.append({'MODEL_NAME': {name}}, ignore_index=True)
        output_df.loc[output_df['MODEL_NAME'] == {name}, "roc_auc_CV_PARAM"] = rmse
        output_df.loc[output_df['MODEL_NAME'] == {name}, "time_minutes_tunning"] = time_taken_minutes
        best_models[name] = model


def model_stacking_manual(df):
    ######################################################
    # # Stacking & Ensemble Learning
    ######################################################
    global train_df, test_df
    df = df.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
    train_df = df[df['TARGET'].notnull()]
    test_df = df[df['TARGET'].isnull()]

    train_df = train_df.fillna(-1)
    filcols = [col for col in test_df.columns if 'TARGET' not in col]
    test_df[filcols] = test_df[filcols].fillna(-1)

    global X, y, X_train, X_test, y_train, y_test
    y = train_df["TARGET"]
    X = train_df.drop(["SK_ID_CURR", "TARGET"], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

    LightGBM = LGBMClassifier(random_state=1, colsample_bytree=0.1, n_estimators=300,
                              learning_rate=0.1)  # local=0.7878  Kaggle=0.79329
    # LightGBM = LGBMClassifier(random_state=1,colsample_bytree= 0.7, max_depth= 3, n_estimators=500) #local=0.7881 kaggle=0.79119
    # LightGBM = LGBMClassifier(nthread=-1,
    #                        # device_type='gpu',
    #                        n_estimators=2872,  # i get this with early stopping.
    #                        learning_rate=0.01,
    #                        max_depth=11,
    #                        num_leaves=58,
    #                        colsample_bytree=0.613,
    #                        subsample=0.708,
    #                        max_bin=407,
    #                        reg_alpha=3.564,
    #                        reg_lambda=4.930,
    #                        min_child_weight=6,
    #                        min_child_samples=165,
    #                        # keep_training_booster=True,
    #                        silent=-1,
    #                        verbose=-1) # roc_auc_cv (voting_LGBM_Cat): 0.7919 kaggle 79228

    # CatBoost = CatBoostClassifier(verbose=False, learning_rate=0.1, iterations=500, depth=6) # local=0.7878  Kaggle=0.79329
    CatBoost = CatBoostClassifier(verbose=False, learning_rate=0.1, iterations=500,
                                  depth=6)  # local=0.7878  Kaggle=0.79329

    # roc_auc(After): 0.7863(CatBoost) CatBoost best params: {'learning_rate': 0.1, 'iterations': 500, 'depth': 6}

    HistGradient = HistGradientBoostingClassifier(max_iter=100, max_depth=None, learning_rate=0.05,
                                                  l2_regularization=0.0)
    models = [
        "voting_LGBM_Cat", "voting_LGBM_HIST", 'voting_Cat_HIST', 'voting_LGBM_CAT_HIST'
                                                                  "stacking_Cat_HIST_LGBM", 'stacking_LGBM_HIST_Cat',
    ]
    classifiers = [
        ("voting_LGBM_Cat",
         VotingClassifier(estimators=[('LightGBM', LightGBM), ('CatBoost', CatBoost)], voting='soft')),
        #
        # ('voting_LGBM_HIST', VotingClassifier(
        #     estimators=[('LightGBM', LightGBM), ('HistGradient', HistGradient)], voting='soft')),
        #
        # ('voting_Cat_HIST', VotingClassifier(
        #     estimators=[('CatBoost', CatBoost), ('HistGradient', HistGradient)], voting='soft')),
        #
        # ('voting_LGBM_CAT_HIST',
        #  VotingClassifier(estimators=[('LightGBM', LightGBM), ('CatBoost', CatBoost)
        #      , ('HistGradient', HistGradient)], voting='soft')),
        #
        # ('stacking_Cat_HIST_LGBM', StackingClassifier(
        #     estimators=[('CatBoost', CatBoost), ('HistGradient', HistGradient)]
        #     , final_estimator=LightGBM, stack_method='predict_proba')),
        #
        # ('stacking_LGBM_HIST_Cat', StackingClassifier(
        #     estimators=[('LightGBM', LightGBM), ('HistGradient', HistGradient)]
        #     , final_estimator=CatBoost, stack_method='predict_proba')),
        #
        # ('stacking_LGBM_Cat_HIST', StackingClassifier(
        #     estimators=[('LightGBM', LightGBM), ('CatBoost', CatBoost)]
        #     , final_estimator=HistGradient, stack_method='predict_proba'))
    ]

    global output_df
    output_df = pd.DataFrame(models, columns=["MODEL_NAME"])
    # output_df.drop('MODEL_BASE', axis=1, inplace=True)

    for name, regressor in classifiers:
        t0 = time.time()
        print("Running Voting/Stacking --> ", name)

        # model = regressor.fit(X_train, y_train)
        # y_prob = model.predict_proba(X_test)[:, 1]
        # roc_score = round(roc_auc_score(y_test, y_prob), 4)
        # print(f"roc_auc_single ({name}):", roc_score)

        model = regressor.fit(X, y)
        roc_score = np.mean((cross_val_score(model, X, y, cv=3, scoring="roc_auc")))
        print(f"roc_auc_cv ({name}): {round(roc_score, 4)} ")
        time_taken_minutes = (time.time() - t0) / 60
        print(f'Total time taken in {name} training: ', round(time_taken_minutes, 2), 'minutes!')
        print("\n")
        output_df = output_df.append({'MODEL_NAME': {name}}, ignore_index=True)
        output_df.loc[output_df['MODEL_NAME'] == {name}, "roc_auc_CV_PARAM"] = roc_score
        output_df.loc[output_df['MODEL_NAME'] == {name}, "time_minutes_tunning"] = time_taken_minutes
        output_df.sort_values(['roc_auc_CV_PARAM'], ascending=False, inplace=True)

        # bestModel = output_df.sort_values(['roc_auc_CV_PARAM'])['MODEL_NAME'][0]
        # Create sub df to submit at kaggle...
        if name == 'voting_LGBM_Cat':
            sub_x = test_df.drop(["SK_ID_CURR", "TARGET"], axis=1)
            test_df['TARGET'] = model.predict_proba(sub_x)[:, 1]
            test_df[['SK_ID_CURR', 'TARGET']].to_csv('submission.csv', index=False)

    return output_df


# -- Score improvements
def final_model(df):
    ######################################################
    # # Stacking & Ensemble Learning
    ######################################################
    global train_df, test_df
    df = df.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
    train_df = df[df['TARGET'].notnull()]
    test_df = df[df['TARGET'].isnull()]

    train_df = train_df.fillna(-1)
    filcols = [col for col in test_df.columns if 'TARGET' not in col]
    test_df[filcols] = test_df[filcols].fillna(-1)

    global X, y, X_train, X_test, y_train, y_test
    y = train_df["TARGET"]
    X = train_df.drop(["SK_ID_CURR", "TARGET"], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)
    ###############################################################
    global model
    LightGBM = LGBMClassifier(random_state=1, colsample_bytree=0.1, n_estimators=1200, learning_rate=0.1, max_depth=6)
    CatBoost = CatBoostClassifier(verbose=False, learning_rate=0.1, iterations=1200, depth=8)
    # estimators = [('LightGBM', LightGBM), ('CatBoost', CatBoost)]
    # model = VotingClassifier(estimators=estimators, voting='soft', weights=[1, 1])
    # # roc_auc score -->   local:     kaggle: 0.79643    csv_blending : 0.79737
    ################################################################

    models = [
        "voting_LGBM_Cat", "voting_LGBM_HIST", 'voting_Cat_HIST', 'voting_LGBM_CAT_HIST'
                                                                  "stacking_Cat_HIST_LGBM", 'stacking_LGBM_HIST_Cat',
    ]

    classifiers = [
        ("voting_LGBM_Cat",
         VotingClassifier(estimators=[('LightGBM', LightGBM), ('CatBoost', CatBoost)], voting='soft')),
    ]

    global output_df
    output_df = pd.DataFrame(models, columns=["MODEL_NAME"])
    # output_df.drop('MODEL_BASE', axis=1, inplace=True)

    for name, regressor in classifiers:
        t0 = time.time()
        print("Running Voting/Stacking --> ", name)

        # model = regressor.fit(X_train, y_train)
        # y_prob = model.predict_proba(X_test)[:, 1]
        # roc_score = round(roc_auc_score(y_test, y_prob), 4)
        # print(f"roc_auc_single ({name}):", roc_score)

        model = regressor.fit(X, y)
        roc_score = np.mean((cross_val_score(model, X, y, cv=3, scoring="roc_auc")))
        print(f"roc_auc_cv ({name}): {round(roc_score, 4)} ")
        time_taken_minutes = (time.time() - t0) / 60
        print(f'Total time taken in {name} training: ', round(time_taken_minutes, 2), 'minutes!')
        print("\n")
        output_df = output_df.append({'MODEL_NAME': {name}}, ignore_index=True)
        output_df.loc[output_df['MODEL_NAME'] == {name}, "roc_auc_CV_PARAM"] = roc_score
        output_df.loc[output_df['MODEL_NAME'] == {name}, "time_minutes_tunning"] = time_taken_minutes
        output_df.sort_values(['roc_auc_CV_PARAM'], ascending=False, inplace=True)

        # bestModel = output_df.sort_values(['roc_auc_CV_PARAM'])['MODEL_NAME'][0]
        # Create sub df to submit at kaggle...
        if name == 'voting_LGBM_Cat':
            sub_x = test_df.drop(["SK_ID_CURR", "TARGET"], axis=1)
            test_df['TARGET'] = model.predict_proba(sub_x)[:, 1]
            test_df[['SK_ID_CURR', 'TARGET']].to_csv('submission.csv', index=False)
    joblib.dump(model, 'final_model.pkl')
    return output_df


def final_model_selected_features(dfin, col_thr=0.1):
    # 0=>415
    # 1=>320
    # 2=>251
    # 3=>204
    # 4=>173
    # tried 0=>(0.79036)  ,  1=>(0.79203)  ,  2=>(0.78913)   ,   3=>(0.79076)
    ######################################
    # Feature SELECTION
    ######################################
    # dfin = df_final.copy()
    df = dfin.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
    train_df = df[df['TARGET'].notnull()]
    test_df = df[df['TARGET'].isnull()].drop("TARGET", axis=1)
    y = train_df["TARGET"]
    X = train_df.drop(["SK_ID_CURR", "TARGET"], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)
    global model
    model = LGBMClassifier(random_state=1).fit(X_train, y_train)
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': X.columns})
    feature_imp = feature_imp.sort_values(by="Value", ascending=False).reset_index(drop=True)
    # feature_imp = feature_imp[0:col_num]
    feature_imp = feature_imp[feature_imp['Value'] > col_thr]

    imp = ["SK_ID_CURR", "TARGET"]
    imp.extend(feature_imp['Feature'].values.tolist())

    print("Number of features to the model : ", len(imp))

    df = df[imp]
    ######################################################
    # # Stacking & Ensemble Learning
    ######################################################
    # df = df.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
    train_df = df[df['TARGET'].notnull()]
    test_df = df[df['TARGET'].isnull()]

    train_df = train_df.fillna(-1)
    filcols = [col for col in test_df.columns if 'TARGET' not in col]
    test_df[filcols] = test_df[filcols].fillna(-1)

    y = train_df["TARGET"]
    X = train_df.drop(["SK_ID_CURR", "TARGET"], axis=1)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)
    ###############################################################
    LightGBM = LGBMClassifier(random_state=1, colsample_bytree=0.1, n_estimators=1200, learning_rate=0.1, max_depth=6)
    CatBoost = CatBoostClassifier(verbose=False, learning_rate=0.1, iterations=1200, depth=8)
    # estimators = [('LightGBM', LightGBM), ('CatBoost', CatBoost)]
    # model = VotingClassifier(estimators=estimators, voting='soft', weights=[1, 1])
    # # roc_auc score -->   local:     kaggle: 0.79643    csv_blending : 0.79737
    ################################################################

    models = [
        "voting_LGBM_Cat", "voting_LGBM_HIST", 'voting_Cat_HIST', 'voting_LGBM_CAT_HIST'
                                                                  "stacking_Cat_HIST_LGBM", 'stacking_LGBM_HIST_Cat',
    ]

    classifiers = [
        ("voting_LGBM_Cat",
         VotingClassifier(estimators=[('LightGBM', LightGBM), ('CatBoost', CatBoost)], voting='soft')),
    ]

    global output_df
    output_df = pd.DataFrame(models, columns=["MODEL_NAME"])
    # output_df.drop('MODEL_BASE', axis=1, inplace=True)

    for name, regressor in classifiers:
        t0 = time.time()
        print("Running Voting/Stacking --> ", name)

        # model = regressor.fit(X_train, y_train)
        # y_prob = model.predict_proba(X_test)[:, 1]
        # roc_score = round(roc_auc_score(y_test, y_prob), 4)
        # print(f"roc_auc_single ({name}):", roc_score)

        model = regressor.fit(X, y)
        roc_score = np.mean((cross_val_score(model, X, y, cv=3, scoring="roc_auc")))
        print(f"roc_auc_cv ({name}): {round(roc_score, 4)} ")
        time_taken_minutes = (time.time() - t0) / 60
        print(f'Total time taken in {name} training: ', round(time_taken_minutes, 2), 'minutes!')
        print("\n")
        output_df = output_df.append({'MODEL_NAME': {name}}, ignore_index=True)
        output_df.loc[output_df['MODEL_NAME'] == {name}, "roc_auc_CV_PARAM"] = roc_score
        output_df.loc[output_df['MODEL_NAME'] == {name}, "time_minutes_tunning"] = time_taken_minutes
        output_df.sort_values(['roc_auc_CV_PARAM'], ascending=False, inplace=True)

        # bestModel = output_df.sort_values(['roc_auc_CV_PARAM'])['MODEL_NAME'][0]
        # Create sub df to submit at kaggle...
        if name == 'voting_LGBM_Cat':
            sub_x = test_df.drop(["SK_ID_CURR", "TARGET"], axis=1)
            test_df['TARGET'] = model.predict_proba(sub_x)[:, 1]
            test_df[['SK_ID_CURR', 'TARGET']].to_csv('submission.csv', index=False)

    return output_df


# -- Presentation documents
def plot_feature_importance_voting(df):  # NOT WORKING
    # df = df_final.iloc[:50,:20].copy()
    df = df.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
    train_df = df[df['TARGET'].notnull()]
    test_df = df[df['TARGET'].isnull()]

    train_df = train_df.fillna(-1)
    filcols = [col for col in test_df.columns if 'TARGET' not in col]
    test_df[filcols] = test_df[filcols].fillna(-1)

    global X, y, X_train, X_test, y_train, y_test
    y = train_df["TARGET"]
    X = train_df.drop(["SK_ID_CURR", "TARGET"], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

    LightGBM = LGBMClassifier(random_state=1, colsample_bytree=0.1, n_estimators=1200, learning_rate=0.1, max_depth=6)
    CatBoost = CatBoostClassifier(verbose=False, learning_rate=0.1, iterations=1200, depth=8)
    estimators = [('LightGBM', LightGBM), ('CatBoost', CatBoost)]
    model = VotingClassifier(estimators=estimators, voting='soft', weights=[1, 1])
    model.fit(X, y)

    feature_imp = pd.DataFrame()
    feature_imp['Features'] = X.columns
    feature_imp['lgbm'] = model.estimators_[0].feature_importances_
    feature_imp['lgbm100'] = model.estimators_[0].feature_importances_ / sum(
        model.estimators_[0].feature_importances_) * 100
    feature_imp['catboost'] = model.estimators_[1].feature_importances_
    feature_imp['catboost100'] = model.estimators_[1].feature_importances_ / sum(
        model.estimators_[1].feature_importances_) * 100
    feature_imp['Voting Importance'] = (feature_imp['lgbm100'] + feature_imp['catboost100']) / 2

    feature_imp = feature_imp.sort_values('Voting Importance', ascending=False).reset_index(drop=True)
    feature_imp.to_csv("feature_importance_Voting.csv", index=False)

    return feature_imp


def plot_roc_auc(df):
    df = df.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
    train_df = df[df['TARGET'].notnull()]
    test_df = df[df['TARGET'].isnull()]

    train_df = train_df.fillna(-1)
    filcols = [col for col in test_df.columns if 'TARGET' not in col]
    test_df[filcols] = test_df[filcols].fillna(-1)

    y = train_df["TARGET"]
    X = train_df.drop(["SK_ID_CURR", "TARGET"], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

    LightGBM = LGBMClassifier(random_state=1, colsample_bytree=0.1, n_estimators=1200, learning_rate=0.1, max_depth=6)
    CatBoost = CatBoostClassifier(verbose=False, learning_rate=0.1, iterations=1200, depth=8)
    estimators = [('LightGBM', LightGBM), ('CatBoost', CatBoost)]
    voting_model = VotingClassifier(estimators=estimators, voting='soft', weights=[1, 1])

    model = wrap(voting_model)
    visualizer = ROCAUC(model, classes=["0", "1"], micro=False, macro=False)
    visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer
    visualizer.score(X_test, y_test)  # Evaluate the model on the test data
    visualizer.show()  # Finalize and show the figure


def plot_pr(df):
    from yellowbrick.classifier import PrecisionRecallCurve
    df = df.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
    train_df = df[df['TARGET'].notnull()]
    test_df = df[df['TARGET'].isnull()]

    train_df = train_df.fillna(-1)
    filcols = [col for col in test_df.columns if 'TARGET' not in col]
    test_df[filcols] = test_df[filcols].fillna(-1)

    global X, y, X_train, X_test, y_train, y_test
    y = train_df["TARGET"]
    X = train_df.drop(["SK_ID_CURR", "TARGET"], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

    LightGBM = LGBMClassifier(random_state=1, colsample_bytree=0.1, n_estimators=1200, learning_rate=0.1, max_depth=6)
    CatBoost = CatBoostClassifier(verbose=False, learning_rate=0.1, iterations=1200, depth=8)
    estimators = [('LightGBM', LightGBM), ('CatBoost', CatBoost)]
    voting_model = VotingClassifier(estimators=estimators, voting='soft', weights=[1, 1])

    model = wrap(voting_model)
    # visualizer = PrecisionRecallCurve(model, classes=["0", "1"], per_class=True)
    visualizer = PrecisionRecallCurve(model, classes=["0", "1"])  # with area
    visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer
    visualizer.score(X_test, y_test)  # Evaluate the model on the test data
    visualizer.show()  # Finalize and show the figure


def plot_classification_report(df, support=True):
    from yellowbrick.classifier import PrecisionRecallCurve
    df = df.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
    train_df = df[df['TARGET'].notnull()]
    test_df = df[df['TARGET'].isnull()]

    train_df = train_df.fillna(-1)
    filcols = [col for col in test_df.columns if 'TARGET' not in col]
    test_df[filcols] = test_df[filcols].fillna(-1)

    global X, y, X_train, X_test, y_train, y_test
    y = train_df["TARGET"]
    X = train_df.drop(["SK_ID_CURR", "TARGET"], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

    LightGBM = LGBMClassifier(random_state=1, colsample_bytree=0.1, n_estimators=1200, learning_rate=0.1, max_depth=6)
    CatBoost = CatBoostClassifier(verbose=False, learning_rate=0.1, iterations=1200, depth=8)
    estimators = [('LightGBM', LightGBM), ('CatBoost', CatBoost)]
    model = VotingClassifier(estimators=estimators, voting='soft', weights=[1, 1])

    from yellowbrick.classifier import ClassificationReport
    viz = ClassificationReport(model, support=support)
    viz.fit(X_train, y_train)
    viz.score(X_test, y_test)
    viz.show()


def plot_model_complexity_lgbm(df):
    print("Running Model Complexity")

    def val_curve_params(model, X, y, param_name, param_range, scoring="roc_auc", cv=3):
        train_score, test_score = validation_curve(model, X=X, y=y, param_name=param_name, param_range=param_range,
                                                   scoring=scoring, cv=cv)

        mean_train_score = np.mean(train_score, axis=1)
        mean_test_score = np.mean(test_score, axis=1)

        plt.plot(param_range, mean_train_score,
                 label="Training Score", color='b')

        plt.plot(param_range, mean_test_score,
                 label="Validation Score", color='g')

        plt.title(f"Validation Curve for LGBM")
        plt.xlabel(f"{param_name}")
        plt.ylabel(f"{scoring}")
        plt.tight_layout()
        plt.legend(loc='best')
        plt.show()

    # LGBM Params
    params = [
        ["learning_rate", [0.01, 0.1, 0.2]],
        # ["n_estimators", [100, 300, 500, 1000, 2000]],
        # ["colsample_bytree", [0.1, 0.3, 0.7, 1]],
        ["max_depth", [4, 6, 8, 10, 12]]
    ]

    # # Catboost Params
    # params = [
    #     ["iterations", [100, 500, 1000, 1200, 2000]],
    #     ["learning_rate", [0.01, 0.1, 0.2]],
    #     ["depth", [4, 6, 8, 10, 12]],
    #     #learning_rate = 0.1, iterations = 1200, depth = 8
    #         ]

    model = LGBMClassifier(random_state=1)
    # model = CatBoostClassifier(random_state=1, verbose=False)
    df = df.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
    train_df = df[df['TARGET'].notnull()]
    test_df = df[df['TARGET'].isnull()]
    train_df = train_df.fillna(-1)
    filcols = [col for col in test_df.columns if 'TARGET' not in col]
    test_df[filcols] = test_df[filcols].fillna(-1)
    y = train_df["TARGET"]
    X = train_df.drop(["SK_ID_CURR", "TARGET"], axis=1)

    for i in range(len(params)):
        val_curve_params(model, X, y, params[i][0], params[i][1])


def plot_model_complexity_cat(df):
    print("Running Model Complexity")

    df = df.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
    train_df = df[df['TARGET'].notnull()]
    test_df = df[df['TARGET'].isnull()]
    train_df = train_df.fillna(-1)
    filcols = [col for col in test_df.columns if 'TARGET' not in col]
    test_df[filcols] = test_df[filcols].fillna(-1)
    y = train_df["TARGET"]
    X = train_df.drop(["SK_ID_CURR", "TARGET"], axis=1)
    y = y.astype(float)
    params = [
        # {"iterations": 100},
        # {"iterations": 500},
        # {"iterations": 1000},
        # {"iterations": 1200},
        # {"iterations": 1500},
        # {"learning_rate": 0.01},
        # {"learning_rate": 0.1},
        # {"learning_rate": 0.2}
        {"depth": 4},
        {"depth": 6},
        {"depth": 8},
        # {"depth": 10},
        # {"depth": 12}
    ]
    test_score = []
    train_score = []
    for i in range(len(params)):
        print(params[i])
        model = CatBoostClassifier(verbose=False)
        model.set_params(**params[i])
        # model.get_params()
        model.fit(X, y)
        result = cross_validate(model, X, y, cv=2, scoring="roc_auc", return_train_score=True)
        test_score.append(np.mean(result['test_score']))
        train_score.append(np.mean(result['train_score']))
    plt.plot([6, 8, 10], train_score, label="Training Score", color='b')
    plt.plot([6, 8, 10], test_score, label="Validation Score", color='g')
    plt.title(f"Validation Curve for CatBoost")
    plt.xlabel(f"depth")
    plt.ylabel(f"roc_auc")
    plt.tight_layout()
    plt.legend(loc='best')
    plt.show()


def plot_feature_importance(df, num=20):
    b = pd.read_csv("feature_importance_Voting.csv")
    plt.figure(figsize=(10, 10))
    sns.barplot(x="Voting Importance", y="Features", data=b[:num])
    plt.title('Feature Importances')
    plt.tight_layout()
    plt.show()


def plot_base_model_comparison():
    b = pd.read_csv("base_models_v1.csv")
    # b = b.sort_values("roc_auc", ascending=True)
    b = b.sort_values("time_minutes", ascending=False)
    fig, ax = plt.subplots()
    x = b['MODEL'].to_list()
    # y = b['roc_auc'].to_list()
    y = b['time_minutes'].to_list()
    width = 0.75
    ind = np.arange(len(y))  # the x locations for the groups
    ax.barh(ind, y)
    ax.set_yticks(ind)
    ax.set_yticklabels(x)
    ax.bar_label(ax.containers[0], fmt='%.2f')
    plt.title('Base Model Training Time (minute)')
    # plt.xlabel('roc_auc')
    plt.xlabel('time in minutes')
    # plt.ylabel('y')
    plt.tight_layout()
    plt.show()


# -- Model Deployment
def flask_model():
    flask_cols = ['NEW_ANNUITY_OVER_CREDIT', 'DAYS_EMPLOYED', 'NEW_EXT_1', 'EXT_SOURCE_2', 'NEW_EXT1_TO_BIRTH_RATIO',
                  'DAYS_LAST_PHONE_CHANGE', 'AMT_GOODS_PRICE', 'AMT_CREDIT', 'DAYS_ID_PUBLISH', 'NEW_AGE_OVER_WORK']
    train = pd.read_csv('datasets/home-credit-default-risk/application_train.csv')
    test = pd.read_csv('datasets/home-credit-default-risk/application_test.csv')
    df = train.append(test).reset_index(drop=True)
    df = df[df['CODE_GENDER'] != 'XNA']
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)
    df['CNT_FAM_MEMBERS'].fillna(0, inplace=True)
    df['DAYS_LAST_PHONE_CHANGE'].replace(0, np.nan, inplace=True)
    cols = ['FLAG_OWN_CAR', 'EMERGENCYSTATE_MODE', 'FLAG_OWN_REALTY']
    for col in cols:
        df[col] = df[col].apply(lambda x: x if x is np.nan else (0 if 'N' in x else 1))
    df['NEW_AGE_OVER_CHLD'] = df['CNT_CHILDREN'] / (df['DAYS_BIRTH'] / 365)
    df['NEW_LIVEAREA_OVER_FAM_MEMBERS'] = df['LIVINGAREA_AVG'] / (df['CNT_FAM_MEMBERS'] + 1)
    cols_to_sum = [col for col in df.columns if "FLAG_DOCUMEN" in col]
    df['NEW_DOCUMENT_FULLFILLMENT'] = df[cols_to_sum].sum(axis=1) / len(cols_to_sum)
    cols_to_sum = [col for col in df.columns if "FLAG_" in col]
    cols_to_sum = [col for col in cols_to_sum if "FLAG_DOC" not in col]
    df['NEW_FLAG_ITEMS'] = df[cols_to_sum].sum(axis=1) / len(cols_to_sum)
    df['NEW_AGE_OVER_WORK'] = df['DAYS_EMPLOYED'] / (df['DAYS_BIRTH'] / 365)
    df['NEW_INCOME_OVER_CREDIT'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']
    df['NEW_ANNUITY_OVER_CREDIT'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
    df['NEW_PROPERTY_TOTAL'] = 0.1 * df['FLAG_OWN_CAR'] + 0.9 * df['FLAG_OWN_REALTY']
    df['NEW_PROPERTY_TOTAL'] = df['AMT_GOODS_PRICE'] + df['AMT_CREDIT']

    # Number of enquiries to Credit Bureau about the client weightage
    df['NEW_MT_REQ_CREDIT_BUREAU'] = 0.3 * df['AMT_REQ_CREDIT_BUREAU_HOUR'] + 0.25 * df[
        'AMT_REQ_CREDIT_BUREAU_DAY'] + \
                                     0.175 * df['AMT_REQ_CREDIT_BUREAU_WEEK'] + 0.125 * df[
                                         'AMT_REQ_CREDIT_BUREAU_MON'] + \
                                     0.1 * df['AMT_REQ_CREDIT_BUREAU_QRT'] + 0.05 * df['AMT_REQ_CREDIT_BUREAU_YEAR']

    # Home overall scoring
    df['NEW_HOME_OVERALL_SCORE'] = df['APARTMENTS_AVG'] * 5 + df['BASEMENTAREA_AVG'] * 2 + df[
        'COMMONAREA_AVG'] * 4 + \
                                   df['ELEVATORS_AVG'] * 1 \
                                   + df['EMERGENCYSTATE_MODE'] * 1 + df['ENTRANCES_AVG'] * 1 + df[
                                       'FLOORSMAX_AVG'] * 2 + \
                                   df['FLOORSMIN_AVG'] * 2 \
                                   + df['LANDAREA_AVG'] * 3 + df['LIVINGAPARTMENTS_AVG'] * 2 + df[
                                       'LIVINGAREA_AVG'] * 3 + \
                                   df['NONLIVINGAPARTMENTS_AVG'] * 1 + df['NONLIVINGAREA_AVG'] * 1 + df[
                                       'YEARS_BUILD_AVG'] * -1

    # EXT Source combinations
    df['NEW_EXT_1'] = df['EXT_SOURCE_2'] / df['EXT_SOURCE_3']
    df['NEW_EXT_2'] = df['EXT_SOURCE_2'] ** df['EXT_SOURCE_3']
    df['NEW_EXT_3'] = df['EXT_SOURCE_2'] / df['EXT_SOURCE_3'] * df['EXT_SOURCE_1']
    df['NEW_EXT_4'] = 2 * df['EXT_SOURCE_2'] + 3 * df['EXT_SOURCE_3'] + df['EXT_SOURCE_1']
    df['NEW_EXT_5'] = 4 * df['EXT_SOURCE_1'] + 2 * df['DAYS_BIRTH'] + 1 * df['AMT_ANNUITY'] + \
                      3 * df['EXT_SOURCE_2'] + 4 * df['AMT_GOODS_PRICE'] + 1.5 * df['DAYS_EMPLOYED']
    df['NEW_EXT_6'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']

    # --------------------------------------------------------------------------
    df['NEW_EXT1_TO_BIRTH_RATIO'] = df['EXT_SOURCE_1'] / (df['DAYS_BIRTH'] / 365)
    df['NEW_EXT3_TO_BIRTH_RATIO'] = df['EXT_SOURCE_3'] / (df['DAYS_BIRTH'] / 365)

    #############################################
    # 3.2 Rare Analyzing & Encoding
    #############################################
    # cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(df)
    # rare_analyser(df, "TARGET", cat_cols)
    rare_cols = ['NAME_INCOME_TYPE', 'NAME_TYPE_SUITE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS',
                 'NAME_HOUSING_TYPE', 'OCCUPATION_TYPE', 'WALLSMATERIAL_MODE', 'HOUSETYPE_MODE',
                 'ORGANIZATION_TYPE']
    for col in rare_cols:
        tmp = df[col].value_counts() / len(df)
        rare_labels = tmp[tmp < 0.05].index
        df[col] = np.where(df[col].isin(rare_labels), 'Rare', df[col])

    #############################################
    # 4. Outliers
    #############################################
    # I have tried different thresholds but everytime score decreased. Tree models is not affected from outliers that much.
    # cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(df)
    # col_out = [col for col in num_cols if col not in ['TARGET', 'SK_ID_CURR'] and df[col].nunique() > 2]
    # for col in col_out:
    #     replace_with_thresholds(df, col, q1=0.05, q3=0.95)

    #############################################
    # 5. Label Encoding
    #############################################
    # cols that will change ['NAME_CONTRACT_TYPE', 'CODE_GENDER', 'NAME_HOUSING_TYPE', 'HOUSETYPE_MODE']
    binary_cols = [col for col in df.columns if df[col].dtype not in [int, float] and len(df[col].unique()) == 2]
    for col in binary_cols:
        label_encoder(df, col)

    #############################################
    # 6. Rare Encoding
    #############################################
    # Applied upper part

    #############################################
    # 7. One-Hot Encoding
    #############################################
    # df = pd.get_dummies(df, dummy_na=True)
    df = pd.get_dummies(df)
    # print("application shape: ", df.shape)

    #############################################
    # 8. Scaling (Best is with MinMax. Tried Robust and Standart too)
    #############################################
    # col_sca = [col for col in df.columns if col not in ['TARGET', 'SK_ID_CURR'] and df[col].nunique() > 2]
    # scaler = MinMaxScaler()
    # df[col_sca] = scaler.fit_transform(df[col_sca])

    df = df.replace([-np.inf, np.inf], np.nan)
    # col_sca = [col for col in df.columns if col not in ['TARGET', 'SK_ID_CURR'] and df[col].nunique() > 2]
    # scaler = MinMaxScaler()
    # df[col_sca] = scaler.fit_transform(df[col_sca])

    # df = df_final.copy()
    flask_cols.append("TARGET")
    flask_cols.append("SK_ID_CURR")
    df = df[flask_cols]
    df = df.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
    train_df = df[df['TARGET'].notnull()]
    test_df = df[df['TARGET'].isnull()]

    train_df = train_df.fillna(-1)
    filcols = [col for col in test_df.columns if 'TARGET' not in col]
    test_df[filcols] = test_df[filcols].fillna(-1)

    y = train_df["TARGET"]
    X = train_df.drop(["SK_ID_CURR", "TARGET"], axis=1)

    LightGBM = LGBMClassifier(random_state=1, colsample_bytree=0.1, n_estimators=1000, learning_rate=0.1, max_depth=6)
    CatBoost = CatBoostClassifier(verbose=False, learning_rate=0.1, iterations=1000, depth=8)
    estimators = [('LightGBM', LightGBM), ('CatBoost', CatBoost)]

    global flask_model
    # flask_model = VotingClassifier(estimators=estimators, voting='soft', weights=[1, 1])
    flask_model = LGBMClassifier(random_state=1, colsample_bytree=0.1, n_estimators=1000, learning_rate=0.1,
                                 max_depth=6)
    flask_model.fit(X, y)

    # Score
    roc_score = np.mean((cross_val_score(flask_model, X, y, cv=3, scoring="roc_auc")))
    print(f"roc_auc_cv: {round(roc_score, 4)} ")
    print("\n")

    # model dump
    # joblib.dump(flask_model, "model_voting_joblib.pkl")
    joblib.dump(flask_model, "model_lgbm_joblib.pkl")
    # pickle.dump(flask_model, open("model_voting_pickle.pkl", 'wb'))
    pickle.dump(flask_model, open("model_lgbm_pickle.pkl", 'wb'))

    # submisson file
    sub_x = test_df.drop(["SK_ID_CURR", "TARGET"], axis=1)
    test_df['TARGET'] = flask_model.predict_proba(sub_x)[:, 1]
    test_df[['SK_ID_CURR', 'TARGET']].to_csv('submission.csv', index=False)


def predict_new(input_array=None):
    # Kullanicidan alacaklarimiz
    # 'NEW_ANNUITY_OVER_CREDIT' # df['NEW_ANNUITY_OVER_CREDIT'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
    # 1- 'AMT_ANNUITY'  #Loan annuity
    # 2- 'AMT_CREDIT'   #Credit amount of the loan
    # 3-'DAYS_EMPLOYED'  #How many days before the application the person started current employment
    # 'NEW_EXT_1 #df['NEW_EXT_1'] = df['EXT_SOURCE_2'] / df['EXT_SOURCE_3']
    # 4- 'EXT_SOURCE_2'   #Normalized score from external data source
    # 5- 'EXT_SOURCE_3'   #Normalized score from external data source
    # df['NEW_EXT1_TO_BIRTH_RATIO'] = df['EXT_SOURCE_1'] / (df['DAYS_BIRTH'] / 365)
    # 6- 'DAYS_BIRTH' # Client's age in days at the time of application
    # 7- 'DAYS_LAST_PHONE_CHANGE'  #How many days before application did client change phone
    # 8- 'AMT_GOODS_PRICE' #For consumer loans it is the price of the goods for which the loan is given
    # 9- DAYS_ID_PUBLISH # How many days before the application did client change the identity document with which he applied for the loan
    # 'NEW_AGE_OVER_WORK'   df['NEW_AGE_OVER_WORK'] = df['DAYS_EMPLOYED'] / (df['DAYS_BIRTH'] / 365)
    # 10- DAYS_EMPLOYED  How many days before the application the person started current employment
    # user_input
    if input_array is None:
        input_array = [24700.5000, 406597.5000, -637, 0.2629, 0.1394, -9461, -1134.0000, 351000.0000, -2120, 0.0830]
    df = pd.DataFrame(np.array([input_array])
                      , columns=['AMT_ANNUITY', 'AMT_CREDIT', 'DAYS_EMPLOYED', 'EXT_SOURCE_2', 'EXT_SOURCE_3',
                                 'DAYS_BIRTH', 'DAYS_LAST_PHONE_CHANGE', 'AMT_GOODS_PRICE', 'DAYS_ID_PUBLISH',
                                 'EXT_SOURCE_1'])

    # Backend
    df['NEW_ANNUITY_OVER_CREDIT'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
    df['NEW_EXT_1'] = df['EXT_SOURCE_2'] / df['EXT_SOURCE_3']
    df['NEW_EXT1_TO_BIRTH_RATIO'] = df['EXT_SOURCE_1'] / (df['DAYS_BIRTH'] / 365)
    df['NEW_AGE_OVER_WORK'] = df['DAYS_EMPLOYED'] / (df['DAYS_BIRTH'] / 365)

    flask_cols = ['NEW_ANNUITY_OVER_CREDIT', 'DAYS_EMPLOYED', 'NEW_EXT_1', 'EXT_SOURCE_2', 'NEW_EXT1_TO_BIRTH_RATIO',
                  'DAYS_LAST_PHONE_CHANGE', 'AMT_GOODS_PRICE', 'AMT_CREDIT', 'DAYS_ID_PUBLISH', 'NEW_AGE_OVER_WORK']

    flask_input = df[flask_cols]
    flask_model = joblib.load("model10v2.pkl")
    prediction = flask_model.predict_proba(flask_input)
    print(prediction[0])


with timer("read pkl"):
    df = combine_files()

final_model(df)
