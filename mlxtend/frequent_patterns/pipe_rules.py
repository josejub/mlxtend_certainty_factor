# mlxtend Machine Learning Library Extensions
# Author: José Juan Ubric <jjuanubric@gmail.com> https://github.com/josejub
#
# License: BSD 3 clause

import pandas as pd
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.base import BaseEstimator, TransformerMixin

from .fpgrowth import fpgrowth
from .association_rules import association_rules

from typing import Union

thresholds =  {
    "support": [0, 1],
    "confidence": [0, 1],
    "lift": [0, np.inf],
    "leverage": [-1, 1],
    "conviction": [0, np.inf],
    "zhangs_metric": [-1, 1],
    "certainty_factor": [-1, 1],
    }

df_columns = [
    "antecedents",
    "consequents",
    "antecedent support",
    "consequent support",
    "support",
    "confidence",
    "lift",
    "leverage",
    "conviction",
    "zhangs_metric",
    "certainty_factor",
    ]

class TransactionEncoder(BaseEstimator, TransformerMixin):
    '''
    Scikit Transformer class to encode transactions from Pandas DataFrames. It takes a "raw" pd.DataFrame (without missing values) and returns a OneHotEncoded sparse pd.Dataframe with as many columns as discrete items in the transaction database.

    Parameters
    ------------
    n_bins: Number of bins to partition numerical columns.
    strategy: Strategy used to define the bins' width. See https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.KBinsDiscretizer.html for options.

    Attributes
    ------------
    data_disc: pd.Dataframe
      Discretized pd.Dataframe. It has the discrete items instead of the raw values for each column.
    '''

    def __init__(self, n_bins: int = 3, strategy: str = "kmeans"):
        if not isinstance(n_bins, int) or n_bins < 2:
            raise ValueError(
                "`n_bins` must be a positive "
                "integer greater or equal to 2`. "
                "Got %s." % n_bins
            )
        
        if strategy not in ["kmeans", "uniform", "quantile"]:
            raise ValueError(
                "`n_bins` must be one of"
                "kmeans, uniform or quantile`. "
                "Got %s. See https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.KBinsDiscretizer.html for more info." % n_bins
            )
        
        self.disc = KBinsDiscretizer(n_bins,strategy=strategy, encode="ordinal")

    def fit(self, data: pd.DataFrame):
        """
        Discretize the pd.DataFrame object. It generates items as <column>=<value> for categorical columns or <column>=<interval> for numeric columns.

        Parameters
        ------------
        data : pd.DataFrame object

        """

        if not isinstance(data, pd.DataFrame):
            raise ValueError("`data` must be a pd.Dataframe. Got %s" % type(data))

        numer = data.select_dtypes("number")
        categ = data.select_dtypes("object")
        
        discretized = pd.DataFrame()
        labels_bas = []
        if not numer.empty:
            self.disc = self.disc.fit(numer)
            dat_disc = pd.DataFrame(self.disc.transform(numer),columns=numer.columns)

            labels_numer = {name: [f"{name}={np.round(arr[i],2)}-{np.round(arr[i+1],2)}" for i in range(len(arr)-1)] for name, arr in zip(self.disc.get_feature_names_out(),self.disc.bin_edges_)}
            labels_bas += [l for name in labels_numer for l in labels_numer[name]]
            # Change numerical values by it's corresponding label.
            for i in labels_numer:
                mapper = dict(zip(sorted(dat_disc[i].unique()),labels_numer[str(i)]))
                dat_disc[i] = dat_disc[i].map(mapper)
            discretized = pd.concat([discretized,dat_disc],axis=1)
        
        if not categ.empty:
            # Add labels as <variable>=value for categorical variables.
            categ_aux = {name: [f"{name}={i}" for i in categ[name]] for name in categ}
            labels_categ = {name: np.unique(categ_aux[name]).tolist() for name in categ}
            labels_bas += [l for name in labels_categ for l in labels_categ[name]]
            # Change values by its corresponding label.
            for i in labels_categ:
                mapper = dict(zip(sorted(categ[i].unique()),labels_categ[i]))
                categ[i] = categ[i].map(mapper)
            
            discretized = pd.concat([discretized,categ],axis=1)
        
        self.data_disc = discretized
        return self
    
    def transform(self, data: pd.DataFrame):
        """
        Computes OneHotEncoding representation of the discretized dataframe.

        Parameters
        ------------
        data : pd.DataFrame object

        """

        if not isinstance(data, pd.DataFrame):
            raise ValueError("`data` must be a pd.Dataframe. Got %s" % type(data))

        self.fit(data)
        dat_disc = self.data_disc
        
        # Return one-hot encoded transactions as a pd.Dataframe.
        onehot_encoded = pd.get_dummies(dat_disc,prefix=[""]*len(dat_disc.columns),prefix_sep="",sparse=True).astype(pd.SparseDtype("bool",False))
        return onehot_encoded
    
    def fit_transform(self, data: pd.DataFrame, y=None):
        """
        Computes OneHotEncoding representation of the discretized dataframe.

        Parameters
        ------------
        data : pd.DataFrame object
        """

        if not isinstance(data, pd.DataFrame):
            raise ValueError("`data` must be a pd.Dataframe. Got %s" % type(data))
        
        self.fit(data)
        dat_disc = self.data_disc
        
        # Return one-hot encoded transactions as a pd.Dataframe.
        onehot_encoded = pd.get_dummies(dat_disc,prefix=[""]*len(dat_disc.columns),prefix_sep="",sparse=True).astype(pd.SparseDtype("bool",False))
        return onehot_encoded

class RuleExtractor(BaseEstimator, TransformerMixin):
    '''
    Scikit Transformer class used as a wrapper to extract rules. It presents antecedent and consequent items as text.

    Parameters
    ------------
    min_support:
      Minimum support for the itemset generation step. Set to 0.1 by default
    metric:
      Evaluation metric used to evaluate rule strength in the rule generation step. set to "confidence" by default.
    min_threshold:
      Minimum threshold for the evaluation metric. Set to 0.8 by default.
    max_len
      Maximum length of generated rules. Set to 15 by default.

    Attributes
    ------------
    min_support:
      Minimum support for the itemset generation step. Set to 0.1 by default
    metric:
      Metric used to evaluate rule strength in the rule generation step. set to "confidence" by default.
    min_threshold:
      Minimum threshold for the evaluation metric. Set to 0.8 by default.
    max_len
      Maximum length of generated rules. Set to 15 by default.
    freq:
      Frequent items generated with FPGrowth.
    rules:
      Rules extracted by association_rules. 
    '''

    def __init__(self, min_support: float = 0.1, metric: str = "confidence", min_threshold: float = 0.8, max_len: int = 15):
        if not isinstance(min_support, float) or not (0 < min_support < 1):
            raise ValueError(
                "`min_support` must be a float "
                "between 0 and 1`. "
                "Got %s." % min_support
            )
        
        if metric not in thresholds.keys():
            raise ValueError(
                "`metric` must be one of "
                "confidence, lift, leverage, conviction, zhangs_metric or certainty_factor`. "
                "Got %s." % metric
            )
        
        if not isinstance(min_threshold, np.number) or not (thresholds[metric][0] < min_threshold < thresholds[metric][1]):
            raise ValueError(
                "`min_threshold` must be a numeric value "
                "between %s and %s`. Got %s." % thresholds[metric][0], thresholds[metric][1], min_threshold
            )
        
        if not isinstance(max_len, int) or max_len < 2:
            raise ValueError(
                "`max_len` must be a integer "
                "greater or equal to 2`. "
                "Got %s." % max_len
            )

        self.min_support = min_support
        self.metric = metric
        self.min_threshold = min_threshold
        self.max_len = max_len
        self.freq = None
        self.rules = None
    
    def fit(self, transac: pd.DataFrame):
        """
        Compute frequent itemsets and association rules from those itemsets.

        Parameters
        ------------
        transac: pd.Dataframe containing the transactions in OneHotEncoded format. It can be the output of TransactionEncoder, for example.

        """
        freq = fpgrowth(transac, min_support=self.min_support, use_colnames=True, max_len=self.max_len)
        rules = association_rules(freq, metric=self.metric, min_threshold=self.min_threshold)

        self.freq = freq
        self.rules = rules

        return self

    def transform(self, transac: pd.DataFrame):
        """
        Codify rules antecedent and consequent as text and return it in pd.DataFrame format.

        Parameters
        ------------
        pd.Dataframe containing the transactions in OneHotEncoded format. It can be the output of TransactionEncoder, for example. 

        """
        rules = self.rules

        # Change antecedents from frozensets to string to enable string-based search
        rules["antecedents"] = rules["antecedents"].apply(lambda x: ', '.join(list(x))).astype("unicode")
        rules["consequents"] = rules["consequents"].apply(lambda x: ', '.join(list(x))).astype("unicode")

        return rules
    
    def fit_transform(self, transac: pd.DataFrame, y=None):
        """
        Compute frequent itemsets and rules and codify antecedent and consequent as text and return it in pd.DataFrame format.

        Parameters
        ------------
        transac: pd.Dataframe containing the transactions in OneHotEncoded format. It can be the output of TransactionEncoder, for example. 

        """
        self.fit(transac)

        rules = self.transform(self.rules)

        return rules

class NegativeItems(BaseEstimator, TransformerMixin):
    '''
    Scikit Transformer class used to generate negative items for a given column (one of the variables in data). It takes a OneHotEncoded pd.DataFrame and return a OneHotEncoded sparse pd.DataFrame with additional columns for negative items codified as <variable>=¬<value>.

    Parameters
    ------------
    columns: 
     Name of the column or columns (given as a list) for which negative items should be generated.
    remove_original:
     ¿Should columns corresponding to the positive items be removed from the results? False by default.

    Attributes
    ------------
    min_support:
      Minimum support for the itemset generation step. Set to 0.1 by default
    metric:
      Metric used to evaluate rule strength in the rule generation step. set to "confidence" by default.
    min_threshold:
      Minimum threshold for the evaluation metric. Set to 0.8 by default.
    max_len
      Maximum length of generated rules. Set to 15 by default.
    freq:
      Frequent items generated with FPGrowth.
    rules:
      Rules extracted by association_rules. 
    '''

    def __init__(self, columns: Union[list, str], remove_original: bool = False):
        if not isinstance(columns, str) or not all([isinstance(value, str) for value in columns]):
            raise ValueError(
                "`columns` must be a string "
                "or list containing strings`. "
                "Got %s." % columns
            )
        
        if not isinstance(remove_original, bool):
            raise ValueError(
                "`remove_original` must one of "
                "True or False`. "
                "Got %s." % remove_original
            )
        
        if isinstance(columns, str):
            columns = [columns]
        self.columns = columns
        self.remove_original = remove_original
    
    def fit(self, data: pd.DataFrame):
        pass

    def transform(self, data: pd.DataFrame):
        """
        Generate the sparse OnehotEncoded Dataframe with negative items.

        Parameters
        ------------
        data: pd.Dataframe containing the transactions in OneHotEncoded format. It can be the output of TransactionEncoderDataframe, for example.
        """

        if not all(data.dtypes == "Sparse[bool, False]") or not all(data.dtypes == "bool"):
            raise ValueError(
                "`data` must be a one-hot encoded"
                "dataframe having all bool or all Sparse[bool, False] dtypes`. "
                "Got %s. dtypes" % data.dtypes.tolist()
            )
        
        data_copy = data.copy()
        for variable in self.columns:
            selected_columns = data.columns[data.columns.str.contains(variable)]

            for name in selected_columns:
                neg_name = name.split("=")[0] + "=¬" + name.split("=")[1]
                data_copy[neg_name] = data[name].map(lambda x: True if not x else False).astype(bool)

        # Fill value must be 0-False, and by default negated columns have more True than False values, so the fill value would be True and raises an error.
        # To fix this simply change True by False in those columns
        cols_with_true_fill = [col for col in data_copy.columns if data_copy[col].dtype == pd.SparseDtype(bool) and data_copy[col].sparse.fill_value]
        data_copy[cols_with_true_fill] = data_copy[cols_with_true_fill].replace({True: False})

        # Convert all columns to type "SparseDtype(bool)" with fill value False
        for col in data_copy.columns:
            if data_copy[col].dtype != pd.SparseDtype(bool):
                data_copy[col] = pd.arrays.SparseArray(data_copy[col], dtype=pd.SparseDtype(bool))

        return data_copy
    
    def fit_transform(self, data: pd.DataFrame, y = None):
        """
        Generate the sparse OnehotEncoded Dataframe with negative items.

        Parameters
        ------------
        data: pd.Dataframe containing the transactions in OneHotEncoded format. It can be the output of TransactionEncoderDataframe, for example.
        """

        negative_onehot = self.transform(data)

        return negative_onehot

class FilterByItem(BaseEstimator, TransformerMixin):
    '''
    Scikit Transformer class used to filter rules in a post-mining step containing items. It allows to set the position in which they should be 

    Parameters
    ------------
    items:
      Items that have to be in the filtered rules. It can be a single string, a list of strings or a list of two lists.
    position:
      Position to search items in. It can be None, "antecedents", "consequents" or both, ["antecedents","consequents"]. Set by default to None.
    length:
      Length of itemsets in given position. It can be a None, int number or a 2-length int list. Set by default to None.

    See documentation for use cases.


    Attributes
    ------------
    items:
      Items that have to be in the filtered rules. It can be a single string, a list of strings or a list of two lists.
    position:
      Position to search items in. It can be None, "antecedents", "consequents" or both, ["antecedents","consequents"]. Set by default to None.
    length:
      Length of itemsets in given position. It can be a None, int number or a 2-length int list. Set by default to None.
    '''

    def __init__(self, items, position = None, length = None):
        if not isinstance(items, (str, list)):
            raise ValueError(
                "`items` must be a string "
                "or a list`. "
                "Got %s." % type(items)
            )
        if (len(items) == 2 and isinstance(items[0], list) and isinstance(items[1], list) and not all([isinstance(value, str) for value in items[0]]) and not all([isinstance(value, str) for value in items[1]])):
            raise ValueError(
                "All items in `items`'s sublists must be strings "
                "Got %s invalid items." % [item for sub in items for item in sub if not isinstance(item, str)]
            )
        if (not all([isinstance(value, list) for value in items]) and not all([isinstance(value, str) for value in items])):
            raise ValueError(
                "All items in `items` must be strings "
                "Got %s invalid items." % [item for item in items if not isinstance(item, str)]
            )
        
        if position not in ["antecedents", "consequents", None]:
            raise ValueError(
                "`position` must be one of"
                "antecedents, consequents or None"
                "Got %s" % position
            )
        
        if not isinstance(length, (int, list)):
            raise ValueError(
                "`lenght` must be a string "
                "or a list`. "
                "Got %s. type" % type(length)
            )
        if (len(length) == 2 and isinstance(length[0], list) and isinstance(length[1], list) and not all([isinstance(value, int) for value in length[0]]) and not all([isinstance(value, int) for value in length[1]])):
            raise ValueError(
                "All items in `length`'s sublists must be integers"
                "Got %s invalid items." % [item for sub in items for item in sub if not isinstance(item, int)]
            )
        if (not all([isinstance(value, list) for value in length]) and not all([isinstance(value, int) for value in length])):
            raise ValueError(
                "All items in `length` must be integers "
                "Got %s invalid items." % [item for item in length if not isinstance(item, int)]
            )
        
        self.items = items
        self.position = position
        self.length = length

    def fit(self, rules):
        pass

    def transform(self, rules: pd.DataFrame):
        """
        Filter rules by set parameters.

        Parameters
        ------------
        data : pd.DataFrame object containing mined association rules.

        """

        if not all([col in rules.columns for col in ["antecedents", "consequents"]]):
            raise ValueError(
                "`rules` must be a "
                "pd.DataFrame containing antecedents and consequents"
                "It lacks %s columns." % [col for col in ["antecedents", "consequents"] if col not in rules.columns]
            )
        
        select = pd.DataFrame()

        # Filter for one item and given length in given position.
        if isinstance(self.length, int) and isinstance(self.items, str):
            mask =  rules[self.position].str.contains(self.items,case=False).to_numpy() & (np.array([len(cons) for cons in rules.consequents.str.split(",")]) == self.length)
            select = rules.loc[mask]

        # Filter by various items and length for given position.
        elif isinstance(self.length, int) and isinstance(self.items, list):
            mask =  rules[self.position].str.contains("|".join(self.items),case=False).to_numpy() & (np.array([len(cons) for cons in rules.consequents.str.split(",")]) == self.length)
            aux = rules.loc[mask]
            select = pd.concat([select,aux],axis=0)

        # Filter by one item and lengths in consequent and antecedent.
        elif isinstance(self.length, list) and isinstance(self.items, str):
            mask =  rules[self.position].str.contains(self.items,case=False).to_numpy() & (np.array([len(cons) for cons in rules.antecedents.str.split(",")]) == self.length[0]) & (np.array([len(cons) for cons in rules.consequents.str.split(",")]) == self.length[1])
            select = rules.loc[mask]

        # Filter by various items in both antecedent and consequent and by lengths in consequent and antecedent
        elif isinstance(self.items, list) and isinstance(self.items[0], list) and isinstance(self.items[1], list) and isinstance(self.length, list):
            # print("List - List")
            items_antec = self.items[0]
            items_conse = self.items[1]

            if isinstance(items_antec, str):
                items_antec = [items_antec]
            if isinstance(items_conse, str):
                items_antec = [items_conse]

            for item_antec in items_antec:
                for item_conse in items_conse:
                    mask =  rules.antecedents.str.contains(item_antec,case=False).to_numpy() & rules.consequents.str.contains(item_conse,case=False).to_numpy() & (np.array([len(cons) for cons in rules.antecedents.str.split(",")]) == self.length[0]) & (np.array([len(cons) for cons in rules.consequents.str.split(",")]) == self.length[1])
                    aux = rules.loc[mask]
                    select = pd.concat([select,aux])
        
        # Filter by various items in a given position and by lengths in antecedent and consequent.
        elif isinstance(self.items, list) and all([isinstance(x,str) for x in self.items]) and isinstance(self.length, list):
            for item in self.items:
                mask =  rules[self.position].str.contains(item,case=False).to_numpy() & (np.array([len(cons) for cons in rules.antecedents.str.split(",")]) == self.length[0]) & (np.array([len(cons) for cons in rules.consequents.str.split(",")]) == self.length[1])
                aux = rules.loc[mask]
                select = pd.concat([select,aux],axis=0)
        
        return select
    
    def fit_transform(self, rules: pd.DataFrame, y = None):
        """
        Filter rules by set parameters.

        Parameters
        ------------
        data : pd.DataFrame object containing mined association rules.

        """

        select = self.transform(rules)
        
        return select

class FilterByValue(BaseEstimator, TransformerMixin):
    '''
    Scikit Transformer class used to filter rules in a post-mining step based on metrics value. 

    Parameters
    ------------
    metric: Metric to filter rules.
    value: value of the metric to filter rules based on
    direction: direction of the comparison. It can be one of "<" or ">".
    order_asc: Should the retrieved rules be ordered based on the metric used to filter? Set by default to None, in which case it does not make any ordering.

    See documentation for use cases.


    Attributes
    ------------
    metric: Metric to filter rules.
    value: value of the metric to filter rules based on
    direction: direction of the comparison. It can be one of "<" or ">".
    order_asc: Should the retrieved rules be ordered based on the emtric used to filter? Set by default to None, in which case it does not make any ordering.
    '''

    def __init__(self, metric: str, value: str, direction: str = None, order_asc: bool = None):
        if metric not in thresholds.keys():
            raise ValueError(
                "`metric` must be one of "
                "confidence, lift, leverage, conviction, zhangs_metric or certainty_factor`. "
                "Got %s." % metric
            )
        
        if not isinstance(value, np.number) or not (thresholds[metric][0] < value < thresholds[metric][1]):
            raise ValueError(
                "`min_threshold` must be a numeric value "
                "between %s and %s`. Got %s. " % thresholds[metric][0], thresholds[metric][1], value
            )
        
        if direction not in [">","<"]:
            raise ValueError(
                "`direction` must be "
                "one of < or >`. Got %s." % direction
            )
        
        if not isinstance(order_asc, (bool,None)):
            raise ValueError(
                "`order_asc` must be a bool "
                "or None`. Got %s." % order_asc
            )

        self.metric = metric
        self.value = value
        self.direction = direction
        self.order_asc = order_asc
    
    def fit(self, rules):
        pass

    def transform(self, rules: pd.DataFrame):
        """
        Filter rules based on selected metric and value.

        Parameters
        ------------
        data : pd.DataFrame object containing mined association rules.

        """

        if self.metric not in rules.columns:
            raise ValueError(
                "`rules` must contain a column with"
                "the selected metric name`. Did not found %s on the DataFrame." % self.metric
            )

        select = pd.DataFrame(columns=rules.columns)
        if  isinstance(self.value, int) or isinstance(self.value, float):
            if self.direction == ">":
                mask = rules[self.metric]>self.value
                select = rules.loc[mask]
            if self.direction == "<":
                mask = rules[self.metric]<self.value 
                select = rules.loc[mask]
        elif isinstance(self.value, list):
            mask = rules[self.metric].between(self.value[0], self.value[1])
            select = rules.loc[mask]
        
        if self.order_asc is not None:
            select = select.sort_values(by=self.metric, ascending=self.order_asc)
        
        return select
    
    def fit_transform(self, rules, y = None):
        """
        Filter rules based on selected metric and value.

        Parameters
        ------------
        data : pd.DataFrame object containing mined association rules.

        """

        select = self.transform(rules)
        
        return select
