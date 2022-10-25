"""Preparation of training and prediction inputs

This script processes the given datasets (baskets.parquet, coupons.parquet)
to construct the input features for the two-level predictive model.

Interim steps in this process are the investigation
of the categorical structure of the product range
with P2V-MAP approach and determination
of substitute-complement relations between product categories
based on the categories' cross-elasticity of demand
estimated from coupon redemption history and category co-occurrences.

Before running this script (as well as the following ones)
the environmental variable PATH_REPO needs to be set

About 2 hours are necessary to run the whole script
"""

import os
import yaml

import numpy as np
import pandas as pd
import scipy.sparse

from gensim.models import Word2Vec
import sklearn.manifold
import sklearn.cluster
import sklearn.linear_model
import pickle


def read_yaml(x):
    """Taken from mlim.exercises.h02.data. Reads config from yaml file"""

    with open(x, "r") as con:
        config = yaml.safe_load(con)
    return config


def write_yaml(x, f):
    """Taken from mlim.exercises.h02.data. Writes a dictionary to yaml file"""
    with open(f, "w") as con:
        yaml.safe_dump(x, con)


def co_occurrences_sparse(x, variable_basket="basket", variable_product="product"):
    """Taken from mlim.exercise.e02-homework-1-3.

    Produces co-occurrence matrix for products/categories
    given dataset of purchase histories
    """

    row = x[variable_basket].values
    col = x[variable_product].values
    dim = (x[variable_basket].max() + 1, x[variable_product].max() + 1)

    basket_product_table = scipy.sparse.csr_matrix(
        (np.ones(len(row), dtype=int), (row, col)),
        shape=dim
    )
    co_occurrences_sparse = basket_product_table.T.dot(basket_product_table)
    co_occurrences_dense = co_occurrences_sparse.toarray()
    return co_occurrences_dense


def create_discount_features(dataset: pd.DataFrame, subst_list: list, compl_list: list):
    """Creates discount-based features used as an input on both model levels

    Implementation requires the dataset to have a column 'basket_id'
    and the lists of substitute/complements categories for each category.
    Returns the dataset with added features
    """

    # the column for the sum of discounts offered to the products of substitute categories
    dataset['disc_subst'] = 0
    # the column for the sum of discounts offered to the products of complement categories
    dataset['disc_compl'] = 0
    # the column for the sum of discounts offered to the products of the category
    dataset['disc_self_cat'] = 0

    for i in range(max(dataset['category']) + 1):
        df = pd.DataFrame(dataset[dataset.category.isin(subst_list[i])].groupby(['basket_id']).sum()['discount'])
        df['category'] = i
        df.rename(columns={'discount': 'discount_subst'}, inplace=True)
        dataset = dataset.merge(df, how='left', on=['basket_id', 'category'])
        dataset['discount_subst'] = dataset['discount_subst'].fillna(0)
        dataset['disc_subst'] += dataset['discount_subst']
        dataset.drop('discount_subst', axis=1, inplace=True)

        df = pd.DataFrame(dataset[dataset.category.isin(compl_list[i])].groupby(['basket_id']).sum()['discount'])
        df['category'] = i
        df.rename(columns={'discount': 'discount_compl'}, inplace=True)
        dataset = dataset.merge(df, how='left', on=['basket_id', 'category'])
        dataset['discount_compl'] = dataset['discount_compl'].fillna(0)
        dataset['disc_compl'] += dataset['discount_compl']
        dataset.drop('discount_compl', axis=1, inplace=True)

        df = pd.DataFrame(dataset[dataset.category == i].groupby(['basket_id']).sum()['discount'])
        df['category'] = i
        df.rename(columns={'discount': 'discount_self'}, inplace=True)
        dataset = dataset.merge(df, how='left', on=['basket_id', 'category'])
        dataset['discount_self'] = dataset['discount_self'].fillna(0)
        dataset['disc_self_cat'] += dataset['discount_self']
        dataset.drop('discount_self', axis=1, inplace=True)

        # the column with sums of discounts offered to other products
    # from the category
    dataset['disc_other_in_cat'] = dataset['disc_self_cat'] - dataset['discount']

    del df

    return dataset


def create_other_features(train_set: pd.DataFrame, baskets: pd.DataFrame, price_list: pd.DataFrame,
                          category_list: pd.DataFrame, category_level=False):
    """Creates various features used as an input on both model levels


    Implementation requires the processed dataset to have a column 'y_product',
    also requires dataframe baskets, product-category map
    and product-price map in the form of pandas dataframes
    Returns two types of the datasets with added features:
    dataset with index week-shopper-product
    and aggregated dataset with index week-shopper-category
    """
    # boolean category_level defines the type of returned dataset
    # and determines the line of operations
    if category_level:
        level = 'category'
    else:
        level = 'product'

    # y_category is label for category-level dataset
    # and input feature for product level dataset
    y_category = train_set.loc[train_set['y_product'] == 1, ['week', 'shopper', 'category', 'y_product']].copy()
    y_category.rename(columns={'y_product': 'y_category'}, inplace=True)
    train_set = train_set.merge(y_category, how='left', on=['week', 'shopper', 'category'])
    train_set['y_category'] = train_set['y_category'].fillna(0)

    if category_level:
        train_set = train_set[['week', 'shopper', 'category', 'disc_subst', 'disc_compl',
                               'disc_self_cat', 'y_category']].drop_duplicates(ignore_index=True)
    else:
        train_set.drop(['disc_subst', 'disc_compl', 'disc_self_cat'], axis=1, inplace=True)

    baskets.sort_values(by=['shopper', level, 'week'], inplace=True)

    # feature 'number of weeks since the last purchase' is created
    baskets['time_since_last_purchase_of_' + level] = 0
    baskets['time_since_last_purchase_of_' + level] = np.where((baskets[level] == baskets[level].shift()) &
                                                               (baskets['shopper'] == baskets['shopper'].shift()),
                                                               baskets['week'] - baskets['week'].shift(), 0)

    train_set = train_set.merge(baskets[['week', level, 'shopper', 'time_since_last_purchase_of_' + level]], how='left',
                                on=['week', level, 'shopper'])

    train_set['y_inv_' + level] = 0
    train_set['y_inv_' + level] = np.where(train_set['y_' + level] == 0, 1, 0)

    train_set.sort_values(by=['shopper', level, 'week'], inplace=True)

    train_set['n_prev_purch'] = train_set.groupby(['shopper', level]).cumsum()['y_' + level]
    train_set['spec_number'] = train_set.groupby(['shopper', level, 'n_prev_purch']).cumsum()['y_inv_' + level]

    train_set.loc[np.isnan(train_set['time_since_last_purchase_of_' + level]),
                  'time_since_last_purchase_of_' + level] = train_set['spec_number']
    train_set.loc[(train_set['n_prev_purch'] == 0), 'time_since_last_purchase_of_' + level] = 0

    train_set.drop(['y_inv_' + level, 'spec_number', 'n_prev_purch'], axis=1, inplace=True)

    # feature 'redemption rate' is created
    baskets['coupon_used'] = 0
    baskets.loc[baskets['discount'] > 0, 'coupon_used'] = 1

    train_set = train_set.merge(baskets[['week', 'shopper', level, 'coupon_used']], how='left',
                                on=['week', 'shopper', level])
    train_set['coupon_used'].fillna(0, inplace=True)

    train_set['coupon_offered'] = 0
    if category_level:
        train_set.loc[train_set['disc_self_cat'] > 0, 'coupon_offered'] = 1
    else:
        train_set.loc[train_set['discount'] > 0, 'coupon_offered'] = 1

    train_set['coupon_previously_offered'] = train_set.groupby(['shopper', level]).cumsum()['coupon_offered']
    train_set['coupon_previously_used'] = train_set.groupby(['shopper', level]).cumsum()['coupon_used']

    train_set['redemption_rate'] = train_set['coupon_previously_used'] / train_set['coupon_previously_offered']
    train_set['redemption_rate'].fillna(0, inplace=True)

    train_set.drop(['coupon_used', 'coupon_offered', 'coupon_previously_offered',
                    'coupon_previously_used'], axis=1, inplace=True)

    # feature 'average purchase frequency' is created
    train_set['avg_purch_freq'] = train_set.groupby(['shopper', level]).cumsum()['y_' + level] / (train_set['week'] + 1)

    train_set['redemption_rate_shift'] = np.where(train_set['redemption_rate'] == 0, 0,
                                                  train_set['redemption_rate'].shift())
    train_set['avg_purch_freq_shift'] = np.where(train_set['avg_purch_freq'] == 0, 0,
                                                 train_set['avg_purch_freq'].shift())

    # feature 'average price of the product in the category'
    # is added
    if category_level:
        price_list = price_list.merge(category_list, how='left', on='product')
        train_set = train_set.merge(pd.DataFrame(price_list.groupby('category').mean()['max_price']).rename(columns={
            'max_price': 'avg_cat_price'}), how='left', on='category')

    return train_set


def create_prediction_set(train_set: pd.DataFrame, category_level=False):
    """Creates the unlabeled dataset for week 90 predictions

    Training dataset processed by functions create_discount_functions()
    and create_other_functions()

    Returns datasets for both category-level and product-level models
    The function must be applied to the corresponding dataset
    AND the corresponding value of boolean 'category_level' must be set
    """

    if category_level:
        level = 'category'
    else:
        level = 'product'

    prediction_set = train_set[train_set['week'] == max(train_set['week'])]

    prediction_set['week'] += 1
    prediction_set['time_since_last_purchase_of_' + level] = np.where(
        prediction_set['time_since_last_purchase_of_' + level] == 0,
        0, prediction_set['time_since_last_purchase_of_' + level] + 1)

    # important - if the product/category was bought on week 89
    # then on week 90 "number of weeks since the last purchase" is 1
    prediction_set.loc[(prediction_set['y_' + level] == 1), 'time_since_last_purchase_of_' + level] = 1

    if category_level:
        prediction_set.drop(['redemption_rate_shift', 'avg_purch_freq_shift', 'disc_subst', 'disc_compl',
                             'disc_self_cat', 'y_' + level], axis=1, inplace=True)
    else:
        prediction_set.drop(['redemption_rate_shift', 'avg_purch_freq_shift',  'discount', 'disc_other_in_cat',
                             'y_category', 'y_' + level], axis=1, inplace=True)

    # columns are renamed so that the names of features
    # in training datasets and prediction datasets are the same
    prediction_set.rename(columns={'redemption_rate': 'redemption_rate_shift',
                                   'avg_purch_freq': 'avg_purch_freq_shift'}, inplace=True)

    return prediction_set


if __name__ == "__main__":


    config = read_yaml("config.yaml")
    path_data = config["path"]
    path_processed = f"{path_data}/processed"
    os.makedirs(path_processed, exist_ok=True)

    coupons = pd.read_parquet(f"{path_data}/coupons.parquet")
    baskets = pd.read_parquet(f"{path_data}/baskets.parquet")

    # creates product - full price map
    max_price = baskets.groupby('product')['price'].max()
    max_price = pd.DataFrame(max_price.rename('max_price'))

    # file prediction_index.parquet from Homework 3
    # is used to construct the index (week-shopper-product)
    # for the product-level training dataset
    prediction_index = pd.read_parquet(f"{path_data}/prediction_index.parquet")
    train_set = pd.concat([prediction_index] * (max(baskets['week']) + 1))
    prod_shoppers = (max(prediction_index['shopper']) + 1) * (max(prediction_index['product']) + 1)
    for i in range(max(baskets['week']) + 1):
        train_set.iloc[i * prod_shoppers:i * prod_shoppers + prod_shoppers, train_set.columns.get_loc('week')] = i
    train_set.reset_index(drop=True, inplace=True)

    # the constructed index is merged with given datasets
    # for further feature construction
    train_set = train_set.merge(baskets, how='left', on=['week', 'shopper', 'product'])
    train_set = train_set.merge(coupons, how='left', on=['week', 'shopper', 'product'])
    train_set = train_set.merge(max_price, how='left', on=['product'])

    # creation of target feature for product-level dataset
    train_set['y_product'] = np.where(np.isnan(train_set['price']), 0, 1)

    train_set['discount'].fillna(0, inplace=True)
    train_set['price'].fillna(train_set['max_price'], inplace=True)

    baskets = baskets.merge(coupons, how='left', on=['week', 'shopper', 'product'])
    baskets['discount'].fillna(0, inplace=True)

    # creates feature 'basket_id'
    # <=> numbers all possible pairs week-shopper
    basket_indices = baskets.groupby(['week', 'shopper']).max()
    basket_indices = basket_indices.reset_index()
    basket_indices['basket_id'] = basket_indices.index
    baskets = baskets.merge(basket_indices[['week', 'shopper', 'basket_id']], on=['week', 'shopper'])
    del basket_indices

    # presents the whole purchase history from dataset 'baskets'
    # as the list of lists
    df_baskets = baskets.groupby('basket_id')['product'].apply(list)
    df_baskets = df_baskets.to_frame()
    df_baskets = df_baskets.reset_index(drop=True)
    df_baskets = df_baskets['product'].values.tolist()

    training_set_word2vec = list()
    for i in range(0, len(df_baskets)):
        full_str = [str(elem) for elem in df_baskets[i]]
        training_set_word2vec.append(full_str)

    # the size of the window for word2vec implementation
    # equals the size of the largest basket minus one
    window_size = max(baskets.groupby('basket_id').count()['product']) - 1

    # word2vec is implemented to obtain multidimensional embeddings
    # of the products based on their co-occurrences in the basket
    model = Word2Vec(training_set_word2vec,
                     window=window_size,
                     **config["word2vec"])

    labels = []
    tokens = []

    for word in model.wv.vocab:
        tokens.append(model[word])
        labels.append(word)

    # t-SNE is applied to embeddings to obtain 2-dimensional
    # informative representation for each product
    tsne_model = sklearn.manifold.TSNE(**config['tsne'])
    new_values = tsne_model.fit_transform(tokens)

    tsne = pd.DataFrame(data={'x': new_values.T[0], 'y': new_values.T[1], 'product': labels})
    tsne['product'] = tsne['product'].astype(int)

    tsne = tsne.sort_values(by='product').reset_index(drop=True)

    # number of product categories has been chosen manually
    # with the visual analysis of the product map
    # Other alternative automated approaches (grid search)
    # can be considered
    n_cat = config['data']['n_cat']

    # configuration for k-means algorithm
    config_scale = {
        "with_mean": True,
        "with_std": True,
    }

    config_kmeans = {
        "n_clusters": n_cat,
        "random_state": 1001,
    }

    xy = tsne[['x', 'y']].values.astype('double')

    indices = []
    for j in range(xy.shape[1]):
        for i in range(j):
            if j > i:
                indices.append((i, j))

    # k-means algorithm is applied to 2-dimensional
    # representations of products to obtain cluster labels for each one
    for ix, iy in indices:
        reduced_data = xy[:, [ix, iy]]

        kmeans = sklearn.cluster.KMeans(**config_kmeans)
        kmeans.fit(reduced_data)
        cluster_labels = kmeans.labels_

        # cluster labels are assigned automatically and randomly
        # by k-means and are used further as category labels of products
        tsne['category'] = cluster_labels

    train_set = train_set.merge(tsne[['product', 'category']], how='left', on='product')
    baskets = baskets.merge(tsne[['product', 'category']], how='left', on='product')

    # the lengthy and space-consuming code piece below
    # determines substitute and complement categories
    # for each of derived product categories

    # It makes use of the matrices of category co-occurrences
    # and investigates how the co-occurrence rate of one category changes
    # when the discount is offered for the product of another category

    list_cat_0_disc_cooc = []
    list_cat_10_disc_cooc = []
    list_cat_20_disc_cooc = []
    list_cat_30_disc_cooc = []
    list_cat_40_disc_cooc = []

    # for-loop below constructs 25x5 matrices of category co-occurrences
    # Each of the matrices includes only the baskets that fulfill certain conditions
    # 4 different discount sizes are considered (10%, 20%, 30%, 40%) to ascertain
    # the consistency of cross-category discount effects

    # each of the produced co-occurrence matrices considers only the baskets
    # where specific category is offered with specific discount or no discount

    for i in range(n_cat):
        basket_disc = pd.DataFrame(baskets.loc[(baskets['category'] == i) & (baskets['discount'] == 0), 'basket_id'])
        basket_disc['cat_0_disc'] = 1
        baskets = baskets.merge(basket_disc, how='left', on='basket_id')
        list_cat_0_disc_cooc.append(co_occurrences_sparse(baskets[baskets['cat_0_disc'] == 1],
                                                          variable_basket="basket_id", variable_product="category"))
        baskets = baskets.drop('cat_0_disc', axis=1)

        basket_disc = pd.DataFrame(baskets.loc[(baskets['category'] == i) & (baskets['discount'] == 10), 'basket_id'])
        basket_disc['cat_10_disc'] = 1
        baskets = baskets.merge(basket_disc, how='left', on='basket_id')
        list_cat_10_disc_cooc.append(co_occurrences_sparse(baskets[baskets['cat_10_disc'] == 1],
                                                           variable_basket="basket_id", variable_product="category"))
        baskets = baskets.drop('cat_10_disc', axis=1)

        basket_disc = pd.DataFrame(baskets.loc[(baskets['category'] == i) & (baskets['discount'] == 20), 'basket_id'])
        basket_disc['cat_20_disc'] = 1
        baskets = baskets.merge(basket_disc, how='left', on='basket_id')
        list_cat_20_disc_cooc.append(co_occurrences_sparse(baskets[baskets['cat_20_disc'] == 1],
                                                           variable_basket="basket_id", variable_product="category"))
        baskets = baskets.drop('cat_20_disc', axis=1)

        basket_disc = pd.DataFrame(baskets.loc[(baskets['category'] == i) & (baskets['discount'] == 30), 'basket_id'])
        basket_disc['cat_30_disc'] = 1
        baskets = baskets.merge(basket_disc, how='left', on='basket_id')
        list_cat_30_disc_cooc.append(co_occurrences_sparse(baskets[baskets['cat_30_disc'] == 1],
                                                           variable_basket="basket_id", variable_product="category"))
        baskets = baskets.drop('cat_30_disc', axis=1)

        basket_disc = pd.DataFrame(baskets.loc[(baskets['category'] == i) & (baskets['discount'] == 40), 'basket_id'])
        basket_disc['cat_40_disc'] = 1
        baskets = baskets.merge(basket_disc, how='left', on='basket_id')
        list_cat_40_disc_cooc.append(co_occurrences_sparse(baskets[baskets['cat_40_disc'] == 1],
                                                           variable_basket="basket_id", variable_product="category"))
        baskets = baskets.drop('cat_40_disc', axis=1)

    del basket_disc

    # in each of the produced matrices, we are interested only
    # in co-occurrences with the discounted product

    for i in range(n_cat):
        list_cat_0_disc_cooc[i] = list_cat_0_disc_cooc[i][i, :]
        list_cat_10_disc_cooc[i] = list_cat_10_disc_cooc[i][i, :]
        list_cat_20_disc_cooc[i] = list_cat_20_disc_cooc[i][i, :]
        list_cat_30_disc_cooc[i] = list_cat_30_disc_cooc[i][i, :]
        list_cat_40_disc_cooc[i] = list_cat_40_disc_cooc[i][i, :]

    # from the resulting lists of arrays, we calculate for each category
    # how the share of the baskets where category A co-occurs
    # with category B in the total number of baskets with category B
    # (this share we refer to as co-occurrence rate)
    # changes if the discount is offered for category B

    # We consider this fraction change to be the metric
    # of category cross elasticity of demand

    cat_0_disc_cooc_fractions = []
    cat_10_disc_cooc_fractions = []
    cat_20_disc_cooc_fractions = []
    cat_30_disc_cooc_fractions = []
    cat_40_disc_cooc_fractions = []

    cat_fract_change_10 = []
    cat_fract_change_20 = []
    cat_fract_change_30 = []
    cat_fract_change_40 = []

    for i in range(n_cat):
        cat_0_disc_cooc_fractions.append(list_cat_0_disc_cooc[i] / list_cat_0_disc_cooc[i][i])
        cat_10_disc_cooc_fractions.append(list_cat_10_disc_cooc[i] / list_cat_10_disc_cooc[i][i])
        cat_20_disc_cooc_fractions.append(list_cat_20_disc_cooc[i] / list_cat_20_disc_cooc[i][i])
        cat_30_disc_cooc_fractions.append(list_cat_30_disc_cooc[i] / list_cat_30_disc_cooc[i][i])
        cat_40_disc_cooc_fractions.append(list_cat_40_disc_cooc[i] / list_cat_40_disc_cooc[i][i])
        cat_fract_change_10.append(cat_10_disc_cooc_fractions[i] - cat_0_disc_cooc_fractions[i])
        cat_fract_change_20.append(cat_20_disc_cooc_fractions[i] - cat_0_disc_cooc_fractions[i])
        cat_fract_change_30.append(cat_30_disc_cooc_fractions[i] - cat_0_disc_cooc_fractions[i])
        cat_fract_change_40.append(cat_40_disc_cooc_fractions[i] - cat_0_disc_cooc_fractions[i])

    cat_fract_change_10 = pd.DataFrame(cat_fract_change_10)
    cat_fract_change_20 = pd.DataFrame(cat_fract_change_20)
    cat_fract_change_30 = pd.DataFrame(cat_fract_change_30)
    cat_fract_change_40 = pd.DataFrame(cat_fract_change_40)

    elast_10 = np.empty((25, 25))
    elast_20 = np.empty((25, 25))
    elast_30 = np.empty((25, 25))
    elast_40 = np.empty((25, 25))

    for i in range(n_cat):
        for j in range(n_cat):
            elast_10[i, j] = (cat_fract_change_10.iloc[i, j] + cat_fract_change_10.iloc[j, i]) / 2
            elast_20[i, j] = (cat_fract_change_20.iloc[i, j] + cat_fract_change_20.iloc[j, i]) / 2
            elast_30[i, j] = (cat_fract_change_30.iloc[i, j] + cat_fract_change_30.iloc[j, i]) / 2
            elast_40[i, j] = (cat_fract_change_40.iloc[i, j] + cat_fract_change_40.iloc[j, i]) / 2

    elast_10 = pd.DataFrame(elast_10)
    elast_20 = pd.DataFrame(elast_20)
    elast_30 = pd.DataFrame(elast_30)
    elast_40 = pd.DataFrame(elast_40)

    max_elast_10 = {}
    max_elast_20 = {}
    max_elast_30 = {}
    max_elast_40 = {}

    for i in range(n_cat):
        cat_list_10 = list(elast_10.loc[i, (elast_10.columns != i)].sort_values(ascending=False).index)
        cat_list_20 = list(elast_20.loc[i, (elast_20.columns != i)].sort_values(ascending=False).index)
        cat_list_30 = list(elast_30.loc[i, (elast_30.columns != i)].sort_values(ascending=False).index)
        cat_list_40 = list(elast_40.loc[i, (elast_40.columns != i)].sort_values(ascending=False).index)

        value_list_10 = elast_10.loc[i, (elast_10.columns != i)].sort_values(ascending=False).values
        value_list_20 = elast_20.loc[i, (elast_20.columns != i)].sort_values(ascending=False).values
        value_list_30 = elast_30.loc[i, (elast_30.columns != i)].sort_values(ascending=False).values
        value_list_40 = elast_40.loc[i, (elast_40.columns != i)].sort_values(ascending=False).values

        dict_t_10 = {}
        dict_t_20 = {}
        dict_t_30 = {}
        dict_t_40 = {}

        for j in range(n_cat - 1):
            dict_t_10[cat_list_10[j]] = value_list_10[j]
            dict_t_20[cat_list_20[j]] = value_list_20[j]
            dict_t_30[cat_list_30[j]] = value_list_30[j]
            dict_t_40[cat_list_40[j]] = value_list_40[j]
        max_elast_10[i] = dict_t_10
        max_elast_20[i] = dict_t_20
        max_elast_30[i] = dict_t_30
        max_elast_40[i] = dict_t_40

    cat_subst_10 = []
    cat_compl_10 = []

    cat_subst_20 = []
    cat_compl_20 = []

    cat_subst_30 = []
    cat_compl_30 = []

    cat_subst_40 = []
    cat_compl_40 = []

    # we consider for each of 4 discounts
    # 5 categories with greatest change (increase) of co-occurrence rate
    # and 5 categories with smallest change or highest decrease of co-occurrence rate

    # If some category is present in the top 5 list for all 4 discounts,
    # This category is determined to be complement

    # If some category is present in the top 5 from below list for all 4 discounts,
    # This category is determined to be substitute

    for i in range(n_cat):
        cat_subst_10.append(list(max_elast_10[i].keys())[19:25])
        cat_compl_10.append(list(max_elast_10[i].keys())[0:5])

        cat_subst_20.append(list(max_elast_20[i].keys())[19:25])
        cat_compl_20.append(list(max_elast_20[i].keys())[0:5])

        cat_subst_30.append(list(max_elast_30[i].keys())[19:25])
        cat_compl_30.append(list(max_elast_30[i].keys())[0:5])

        cat_subst_40.append(list(max_elast_40[i].keys())[19:25])
        cat_compl_40.append(list(max_elast_40[i].keys())[0:5])

    cat_subst = []
    cat_compl = []

    for i in range(n_cat):
        cat_subst.append(list(set(cat_subst_10[i]) & set(cat_subst_20[i])
                              & set(cat_subst_30[i]) & set(cat_subst_40[i])))
        cat_compl.append(list(set(cat_compl_10[i]) & set(cat_compl_20[i])
                              & set(cat_compl_30[i]) & set(cat_compl_40[i])))

    # creates feature 'basket_id' for the training set
    basket_indices = train_set.groupby(['week', 'shopper']).max()
    basket_indices = basket_indices.reset_index()
    basket_indices['basket_id'] = basket_indices.index
    train_set = train_set.merge(basket_indices[['week', 'shopper', 'basket_id']], on=['week', 'shopper'])
    del basket_indices

    # saves the lists of substitute/complement categories
    # which are used on a later stage of the pipeline
    with open(f"{path_processed}/cat_subst.txt", 'wb') as fp:
        pickle.dump(cat_subst, fp)

    with open(f"{path_processed}/cat_compl.txt", 'wb') as fp:
        pickle.dump(cat_compl, fp)

    # the function is used to create feature based on discounts
    # and cross-category relations
    train_set = create_discount_features(dataset=train_set, subst_list=cat_subst, compl_list=cat_compl)

    # the function is used to create category-level dataset
    # with category-specific features
    train_set_cat = create_other_features(train_set=train_set, baskets=baskets,
                                          price_list=max_price, category_list=tsne[['product', 'category']],
                                          category_level=True)
    train_set_cat.sort_values(by=['week', 'shopper', 'category'], inplace=True)

    # input and labels for category-level model
    y_cat = pd.DataFrame(train_set_cat['y_category'])
    order_features_cat = ['avg_cat_price', 'disc_self_cat', 'disc_subst',
                          'disc_compl', 'time_since_last_purchase_of_category',
                          'avg_purch_freq_shift', 'redemption_rate_shift']
    X_cat = train_set_cat[order_features_cat]

    # the function is used to create product-level dataset
    # with product-specific features
    train_set_prod = create_other_features(train_set=train_set, baskets=baskets,
                                           price_list=max_price, category_list=tsne[['product', 'category']])
    train_set_prod.sort_values(by=['week', 'shopper', 'product'], inplace=True)

    # input and labels for product-level model
    y_prod = pd.DataFrame(train_set_prod['y_product'])
    order_features_prod = ['max_price', 'discount', 'disc_other_in_cat',
                           'time_since_last_purchase_of_product', 'avg_purch_freq_shift',
                           'redemption_rate_shift', 'y_category']
    X_prod = train_set_prod[order_features_prod]

    # prediction sets for both model levels are created
    prediction_set_cat = create_prediction_set(train_set_cat, category_level=True)
    prediction_set_prod = create_prediction_set(train_set_prod)

    # all necessary processed data is saved
    train_set_cat.to_parquet(f"{path_processed}/train_set_cat.pt")
    train_set_prod.to_parquet(f"{path_processed}/train_set_prod.pt")
    prediction_set_cat.to_parquet(f"{path_processed}/prediction_set_cat.pt")
    prediction_set_prod.to_parquet(f"{path_processed}/prediction_set_prod.pt")
    X_cat.to_parquet(f"{path_processed}/x_cat.pt")
    y_cat.to_parquet(f"{path_processed}/y_cat.pt")
    X_prod.to_parquet(f"{path_processed}/x_prod.pt")
    y_prod.to_parquet(f"{path_processed}/y_prod.pt")
