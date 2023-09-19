from .fpgrowth import fpgrowth
from .association_rules import association_rules

from sklearn.preprocessing import KBinsDiscretizer
import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin

class TransactionEncoderDataframe(BaseEstimator, TransformerMixin):
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
    def __init__(self, n_bins=3,strategy="kmeans"):
        self.disc = KBinsDiscretizer(n_bins,strategy=strategy, encode="ordinal")

    def fit(self, data):
        """
        Discretize the pd.DataFrame object. It generates items as <column>=<value> for categorical columns or <column>=<interval> for numeric columns.

        Parameters
        ------------
        data : pd.DataFrame object

        """
        numer = data.select_dtypes("number")
        categ = data.select_dtypes("object")
        
        discretized = pd.DataFrame()
        labels_bas = []
        if not numer.empty:
            self.disc = self.disc.fit(numer)
            dat_disc = pd.DataFrame(self.disc.transform(numer),columns=numer.columns)

            labels_numer = {name: [f"{name}={np.round(arr[i],2)}-{np.round(arr[i+1],2)}" for i in range(len(arr)-1)] for name, arr in zip(self.disc.get_feature_names_out(),self.disc.bin_edges_)}
            labels_bas += [l for name in labels_numer for l in labels_numer[name]]
            # print(labels_numer)
            # Cambio cada valor entero por la etiqueta que le corresponde.
            for i in labels_numer:
                mapper = dict(zip(sorted(dat_disc[i].unique()),labels_numer[str(i)]))
                # print(mapper)
                dat_disc[i] = dat_disc[i].map(mapper)
            discretized = pd.concat([discretized,dat_disc],axis=1)
        
        if not categ.empty:
            # Añado <Variable> = valor para las variables categóricas.
            categ_aux = {name: [f"{name}={i}" for i in categ[name]] for name in categ}
            labels_categ = {name: np.unique(categ_aux[name]).tolist() for name in categ}
            labels_bas += [l for name in labels_categ for l in labels_categ[name]]
            # print(labels_categ)
            # cambio los valores por las categorías que le corresponda
            for i in labels_categ:
                mapper = dict(zip(sorted(categ[i].unique()),labels_categ[i]))
                # print(mapper)
                categ[i] = categ[i].map(mapper)
            
            discretized = pd.concat([discretized,categ],axis=1)
        
        self.data_disc = discretized
        return self
    
    def transform(self, data):
        """
        Computes OneHotEncoding representation of the discretized dataframe.

        Parameters
        ------------
        data : pd.DataFrame object

        """

        self.fit(data)
        dat_disc = self.data_disc
        
        # Devuelvo un dataframe con las transacciones codificadas como one-hot.
        return(pd.get_dummies(dat_disc,prefix=[""]*len(dat_disc.columns),prefix_sep="",sparse=True)).astype(pd.SparseDtype("bool",False))
    
    def fit_transform(self,data, y=None):
        """
        Computes OneHotEncoding representation of the discretized dataframe.

        Parameters
        ------------
        data : pd.DataFrame object

        """
        self.fit(data)
        dat_disc = self.data_disc
        
        # Devuelvo un dataframe con las transacciones codificadas como one-hot.
        return(pd.get_dummies(dat_disc,prefix=[""]*len(dat_disc.columns),prefix_sep="",sparse=True)).astype(pd.SparseDtype("bool",False))

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
    def __init__(self, min_support=0.1, metric = "confidence", min_threshold = 0.8, max_len = 15):
        self.min_support = min_support
        self.metric = metric
        self.min_threshold = min_threshold
        self.max_len = max_len
        self.freq = None
        self.rules = None
    
    def fit(self, transac):
        """
        Compute frequent itemsets and association rules from those itemsets.

        Parameters
        ------------
        transac: pd.Dataframe containing the transactions in OneHotEncoded format. It can be the output of TransactionEncoder, for example.

        """
        freq = fpgrowth(transac,min_support=self.min_support,use_colnames=True,max_len = self.max_len)
        rules = association_rules(freq,metric=self.metric, min_threshold=self.min_threshold)

        self.freq = freq
        self.rules = rules

        return self

    def transform(self, transac):
        """
        Codify rules antecedent and consequent as text and return it in pd.DataFrame format.

        Parameters
        ------------
        pd.Dataframe containing the transactions in OneHotEncoded format. It can be the output of TransactionEncoder, for example. 

        """
        rules = self.rules
        # Cambio los objetos de consecuente y antecedente de frozenset a string para poder buscar
        rules["antecedents"] = rules["antecedents"].apply(lambda x: ', '.join(list(x))).astype("unicode")
        rules["consequents"] = rules["consequents"].apply(lambda x: ', '.join(list(x))).astype("unicode")

        return rules
    
    def fit_transform(self, transac, y=None):
        """
        Compute frequent itemsets and rules and codify antecedent and consequent as text and return it in pd.DataFrame format.

        Parameters
        ------------
        transac: pd.Dataframe containing the transactions in OneHotEncoded format. It can be the output of TransactionEncoder, for example. 

        """
        self.fit(transac)

        rules = self.rules
        # Cambio los objetos de consecuente y antecedente de frozenset a string para poder buscar
        rules["antecedents"] = rules["antecedents"].apply(lambda x: ', '.join(list(x))).astype("unicode")
        rules["consequents"] = rules["consequents"].apply(lambda x: ', '.join(list(x))).astype("unicode")

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
    def __init__(self, columns, remove_original = False):
        if isinstance(columns, str):
            columns = [columns]
        self.columns = columns
        self.remove_original = remove_original
    
    def fit(self, data):
        pass

    def transform(self, data):
        """
        Generate the sparse OnehotEncoded Dataframe with negative items.

        Parameters
        ------------
        data: pd.Dataframe containing the transactions in OneHotEncoded format. It can be the output of TransactionEncoderDataframe, for example.
        """
        data_copy = data.copy()
        for variable in self.columns:
            selected_columns = data.columns[data.columns.str.contains(variable)]

            for name in selected_columns:
                neg_name = name.split("=")[0] + "=¬" + name.split("=")[1]
                data_copy[neg_name] = data[name].map(lambda x: True if not x else False).astype(bool)

        # Esta parte ha sido necesaria pq el fill value tiene que ser 0-False, y por defecto las columnas negadas tienen más valores True que false, 
        # por lo que el fill value sería True y da error.
        cols_with_true_fill = [col for col in data_copy.columns if data_copy[col].dtype == pd.SparseDtype(bool) and data_copy[col].sparse.fill_value]
        data_copy[cols_with_true_fill] = data_copy[cols_with_true_fill].replace({True: False})

        # Convertir todas las columnas a tipo "SparseDtype(bool)" con fill value False
        for col in data_copy.columns:
            if data_copy[col].dtype != pd.SparseDtype(bool):
                data_copy[col] = pd.arrays.SparseArray(data_copy[col], dtype=pd.SparseDtype(bool))


        return data_copy
    
    def fit_transform(self, data, y = None):
        """
        Generate the sparse OnehotEncoded Dataframe with negative items.

        Parameters
        ------------
        data: pd.Dataframe containing the transactions in OneHotEncoded format. It can be the output of TransactionEncoderDataframe, for example.
        """
        data_copy = data.copy()
        for variable in self.columns:
            selected_columns = data.columns[data.columns.str.contains(variable)]

            for name in selected_columns:
                neg_name = name.split("=")[0] + "=¬" + name.split("=")[1]
                data_copy[neg_name] = data[name].map(lambda x: True if not x else False).astype(bool)

        # Esta parte ha sido necesaria pq el fill value tiene que ser 0-False, y por defecto las columnas negadas tienen más valores True que false, 
        # por lo que el fill value sería True y da error.
        cols_with_true_fill = [col for col in data_copy.columns if data_copy[col].dtype == pd.SparseDtype(bool) and data_copy[col].sparse.fill_value]
        data_copy[cols_with_true_fill] = data_copy[cols_with_true_fill].replace({True: False})

        # Convertir todas las columnas a tipo "SparseDtype(bool)" con fill value False
        for col in data_copy.columns:
            if data_copy[col].dtype != pd.SparseDtype(bool):
                data_copy[col] = pd.arrays.SparseArray(data_copy[col], dtype=pd.SparseDtype(bool))

        return data_copy

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
        self.items = items
        self.position = position
        self.length = length

    def fit(self, rules):
        pass

    def transform(self, rules):
        """
        Filter rules by set parameters.

        Parameters
        ------------
        data : pd.DataFrame object containing mined association rules.

        """
        select = pd.DataFrame()

        # Filter for one item and given length in given position.
        if isinstance(self.length, int) and isinstance(self.items, str):
            # print("Entero - string")
            mask =  rules[self.position].str.contains(self.items,case=False).to_numpy() & (np.array([len(cons) for cons in rules.consequents.str.split(",")]) == self.length)
            select = rules.loc[mask]

        # Filter by various items and length for given position.
        elif isinstance(self.length, int) and isinstance(self.items, list):
            # print("Entero - list")
            mask =  rules[self.position].str.contains("|".join(self.items),case=False).to_numpy() & (np.array([len(cons) for cons in rules.consequents.str.split(",")]) == self.length)
            aux = rules.loc[mask]
            select = pd.concat([select,aux],axis=0)
        # Filter by one item and lengths in consequent and antecedent.
        elif isinstance(self.length, list) and isinstance(self.items, str):
            # print("List - string")
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
    
    def fit_transform(self, rules, y = None):
        """
        Filter rules by set parameters.

        Parameters
        ------------
        data : pd.DataFrame object containing mined association rules.

        """
        select = pd.DataFrame()

        # Filter for one item and given length in given position.
        if isinstance(self.length, int) and isinstance(self.items, str):
            # print("Entero - string")
            mask =  rules[self.position].str.contains(self.items,case=False).to_numpy() & (np.array([len(cons) for cons in rules.consequents.str.split(",")]) == self.length)
            select = rules.loc[mask]

        # Filter by various items and length for given position.
        elif isinstance(self.length, int) and isinstance(self.items, list):
            # print("Entero - list")
            mask =  rules[self.position].str.contains("|".join(self.items),case=False).to_numpy() & (np.array([len(cons) for cons in rules.consequents.str.split(",")]) == self.length)
            aux = rules.loc[mask]
            select = pd.concat([select,aux],axis=0)
        # Filter by one item and lengths in consequent and antecedent.
        elif isinstance(self.length, list) and isinstance(self.items, str):
            # print("List - string")
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

class FilterByValue(BaseEstimator, TransformerMixin):
    '''
    Scikit Transformer class used to filter rules in a post-mining step based on metrics value. 

    Parameters
    ------------
    metric
    value
    direction
    order_asc

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
    def __init__(self, metric, value, direction=None, order_asc=None):
        self.metric = metric
        self.value = value
        self.direction = direction
        self.order_asc = order_asc
    
    def fit(self, rules):
        pass

    def transform(self, rules):
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
