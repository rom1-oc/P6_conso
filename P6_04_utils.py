""" Utils """

import os
import math
import warnings
import math
import nltk
import missingno as msno
import numpy as np
import pandas as pd
import seaborn as sns
import pyLDAvis
import imp 
import cv2
import tensorflow as tf
import plotly.express as px
import tensorflow.experimental.numpy as tnp
#import statsmodels.api as sm
from pyLDAvis import sklearn
from datetime import datetime
from matplotlib import pyplot as plt
from hyperopt import hp, fmin, tpe, hp, anneal, Trials, STATUS_OK
from sklearn import preprocessing, decomposition, metrics
from string import ascii_letters
from PIL import Image, ImageOps, ImageFilter
from keras import Model
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor, make_column_selector
from sklearn.pipeline import Pipeline, make_pipeline                          
from sklearn.impute import SimpleImputer                   
from sklearn.experimental import enable_iterative_imputer
from sklearn.decomposition import PCA, LatentDirichletAllocation
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, StratifiedKFold, RepeatedStratifiedKFold, GridSearchCV, learning_curve
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, QuantileTransformer, RobustScaler, MinMaxScaler, normalize
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import accuracy_score, silhouette_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.manifold import TSNE
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.metrics.cluster import adjusted_rand_score
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import SGD
from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from yellowbrick.features import RadViz
from yellowbrick.datasets import load_concrete
from yellowbrick.cluster import KElbowVisualizer


from scipy.sparse import * 



def info(dataframe):
    """Prints dataframe parameters
    
    Args:
        dataframe (pd.Dataframe): data source
    Returns:
        Prints parameters: number of columns, number of rows, rate of missing values
    """
    print(str(len(dataframe.columns.values)) + " columns" )
    print(str(len(dataframe)) + " rows")
    print("Rate of missing values in df : " + str(dataframe.isnull().mean().mean()*100) + " %")
    print("")
    
    
def inter_quartile_method_function(dataframe):
    """Remove data outliers with inter-quartile method
    
    Args:
        dataframe (pd.Dataframe): input
    Returns:
        dataframe (pd.Dataframe): output
    """
    #
    q1 = dataframe.quantile(0.25)
    q3 = dataframe.quantile(0.75)
    iqr = q3 - q1
    #
    dataframe = dataframe[(dataframe <= dataframe.quantile(0.75) + 1.5*iqr)
                                                        & (dataframe >= dataframe.quantile(0.25) - 1.5*iqr)]

    return dataframe


def inferior_to_max_quantile_function(dataframe, quantile):
    """Remove data outliers superior to given quantile 
    
    Args:
        dataframe (pd.Dataframe): input
        quantile (float):
    Returns:
        dataframe (pd.Dataframe): output
    """
    maximum = dataframe.quantile(quantile) 
    dataframe = dataframe[dataframe < maximum]
    
    return dataframe


def image_params_function(image):
    """Display image parameters
    
    Args:
        image (PIL.PngImagePlugin.PngImageFile): input
    Returns:
        -
    """
    # Display size of image
    w, h = image.size
    print("Largeur : {} px, hauteur : {} px".format(w, h))
    
    # Display image quantification mode
    print("Format des pixels : {}".format(image.mode))
    
    # Display pixels matrix shape
    mat = np.array(image)
    print("Taille de la matrice de pixels : {}".format(mat.shape))
    

def prepare_category_column_function(dataframe, feature, depth):
    """xxx
    
    Args:
        dataframe (pd.Dataframe): input
        feature (str): xxx
        depth (int): xxx
    Returns:
        dataframe (pd.Dataframe): output
    """
    # Prune category column
    # Selecting first three sub-categories only
    dataframe['product_category'] = dataframe[feature].apply(lambda x: x.replace('["', ''))
    dataframe['product_category'] = dataframe['product_category'].apply(lambda x: x.replace('"]', ''))
    dataframe['product_category'] = dataframe['product_category'].apply(lambda x: x.split(">>")[:depth])
    
    return dataframe


def normalize_text(text, stopwords, normalization_type):
    """Tokenizes, removes punctuation, removes stopwords, stems words
    
    Args:
        text (String): text to normalize
        stopwords List(String): list of stop words
        normalization_type (String): stemming or lemming
    Returns:
         tokens (String): normalized text
    """
    # lower case
    text = text.lower()
    
    # tokenize by removing punctuation
    tokenizer = nltk.RegexpTokenizer(r'[A-Za-z]+')
    tokens = tokenizer.tokenize(text)

    # remove stopwords
    tokens = [t for t in tokens if t not in stopwords]
          
    if normalization_type == 'stemming':  
        
        # stem words  
        stemmer = PorterStemmer()

        for i in range(len(tokens)):
            tokens[i] = stemmer.stem(tokens[i])

    elif normalization_type == 'lemming':    
        
        # lem words  
        lemmatizer = WordNetLemmatizer()
        
        for i in range(len(tokens)):
            tokens[i] = lemmatizer.lemmatize(tokens[i])
            
    # convert description lists to strings
    string_ = " ".join(tokens)
    
    return string_


def preprocessing_description_function(dataframe, feature, stopwords):
    """Tokenizes, removes punctuation, removes stopwords, stems words
    
    Args:
        dataframe (pd.Dataframe): data source
        feature (String): list of stop words
        stopwords List(String): list of stop words 
        normalization_type (String): stemming or lemming
    Returns:
        dataframe (pd.Dataframe): output
    """
    
    #
    dataframe = dataframe[['product_name',
                           'product_category',
                           feature]].copy()
    
    normalization_types = ['stemming', 'lemming']
    
    for normalization_type in normalization_types:
        #
        new_feature = (feature + '_normalize_' + normalization_type)
        dataframe[new_feature] = dataframe[feature].apply(lambda x: normalize_text(x, stopwords, normalization_type))

    return dataframe
    

def select_category_depth_function(dataframe, depth):
    """Pick the category level to consider (levels are separated by ">>")
    
    Args:
        dataframe (pd.Dataframe): data source
        depth (int): level of categories to consider
    Returns:
        dataframe (pd.Dataframe): output
    """
    # Select categories at level depth 
    dataframe["product_category"] = dataframe["product_category"].apply(lambda x: x[depth-1])
    
    
    n = len(dataframe["product_category"].unique())
    print("Unique categories at level " + str(depth) + ": " + str(n) + "\n")
        
    return dataframe


def tfidf_function(dataframe, feature, category_filter, labels, params):
    """TfidfVectorizer visualized with TSNE-2d
    
    Args:
        dataframe (pd.Dataframe): data source
        feature (String): column to apply TfidfVectorizer
        category_filter List(String): filter by category type in product_category
        labels List(String): Column of labels to display
        params (dict): parameters of TfidfVectorizer
    Returns:
        dataframe (pd.Dataframe): output
    """
    #
    if category_filter:
        df = dataframe[dataframe.product_category == category_filter].copy()
    else:
        df = dataframe.copy()

    # Overview
    #info(df)

    # tf-idf
    tfidf = TfidfVectorizer(**params)
    X = tfidf.fit_transform(df[feature])
    values = tfidf.get_feature_names_out()

    # t-sne
    tsne = TSNE(n_components=2, random_state=0)
    tsne_results = tsne.fit_transform(X) 

    # Plot
    df['tsne-2d-one'] = tsne_results[:,0]
    df['tsne-2d-two'] = tsne_results[:,1]

    plt.figure(figsize=(16,10))

    f = sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue=labels,
        palette=sns.color_palette("hls", len(dataframe[labels].unique())),
        data=df,
        legend="brief",
        alpha=1)

    # Put the legend out of the figure
    f.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title("T-SNE 2d - " + str(feature) + " - " + category_filter)

    plt.show()
    
    
def word_embedding_function(dataframe, feature, category_filter, max_vocab_size):
    """TfidfVectorizer visualized with TSNE-2d
    
    Args:
        dataframe (pd.Dataframe): data source
        feature (String): column to apply TfidfVectorizer
        category_filter List(String): filter by category type in product_category
        labels List(String): Column of labels to display
        params (dict): parameters of TfidfVectorizer
    Returns:
        dataframe (pd.Dataframe): output
    """
    #
    if category_filter:
        df = dataframe[dataframe.product_category == category_filter].copy()
    else:
        df = dataframe.copy()

    # Overview
    info(df)

    #
    df[feature] = df[feature].apply(lambda x: x.split(" "))
    sentences = df[feature]

    # instantiate the model and fit
    word2vec_model = Word2Vec(sentences, 
                              min_count=1,
                              max_vocab_size=max_vocab_size,
                             )

    # get the vocabulary
    vocab = word2vec_model.wv.key_to_index 
    print("Number of words for " + category_filter + ": " + str(len(vocab)) + " (Word2vec)\n")
    
    # data to transform
    X = word2vec_model.wv[vocab]
    
    # t-sne
    tsne = TSNE(n_components=2, random_state=0)
    results = tsne.fit_transform(X) 

    #####
    # Plot 
    column_names = ["tsne-2d-one", "tsne-2d-two", "vocab"]
    df_plot = pd.DataFrame(columns = column_names)
    df_plot['tsne-2d-one'] = results[:,0]
    df_plot['tsne-2d-two'] = results[:,1]
    df_plot['vocab'] = vocab
    
    plt.figure(figsize=(16,10))

    f = sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="vocab",
        palette=sns.color_palette("hls", len(vocab)),
        data=df_plot,
        legend="brief",
        alpha=1)

    # Put the legend out of the figure
    f.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title("T-SNE 2d - " + str(feature) + " - " + category_filter)
    #####
    
    '''
    # plot the data matplotlib
    fig, ax = plt.subplots(figsize=(15, 10))
    plt.scatter(results[:, 0], results[:, 1])
    ax.set_xlabel("tsne-2d-one")
    ax.set_ylabel("tsne-2d-two")
    ax.set_title("Gensim - Word2Vec - PCA n=2")

    # annotate the plot
    for i, word in enumerate(vocab):
        plt.annotate(word, xy=(results[i, 0], results[i, 1]))'''

    plt.show
    
    return
            
        
def display_topics(model, feature_names, no_top_words):
    """xxx
    
    Args:
        model: , 
        feature_names list(str): , 
        no_top_words (int): number of words by topic
    Returns:
        -
    """
    #
    for topic_idx, topic in enumerate(model.components_):
        print ("Top " + str(no_top_words) + " words for Topic #" + str(topic_idx))
        print (" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]) + "\n")
        
    return


def lda_function(dataframe, feature, category_filter, n_topics, params):
    """TfidfVectorizer 
    
    Args:
        dataframe (pd.Dataframe): data source
        feature (String): column to apply TfidfVectorizer
        category_filter List(String): filter by category type in product_category
        n_topics int: number of topics to find
    Returns:
        -
    """
    #
    if category_filter:
        df = dataframe[dataframe.product_category == category_filter].copy()
    else:
         df = dataframe.copy()
    
    #
    n_topics = n_topics
    documents = df[feature]

    # TF-IDF
    #tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=1000, stop_words='english')
    tf_vectorizer = TfidfVectorizer(**params)
    tf = tf_vectorizer.fit_transform(documents)

    # Create LDA model
    lda = LatentDirichletAllocation(
            n_components=n_topics, 
            max_iter=5, 
            learning_method='online', 
            learning_offset=50.,
            random_state=0)

    # Filter on data
    lda_output = lda.fit_transform(tf)

    #
    print("(" + feature + "):\n")
    display_topics(lda, tf_vectorizer.get_feature_names(), 10)

    return lda, tf, tf_vectorizer


def text_classification_function(dataframe, model, x_feature, y_feature, params):
    """General function for model creation and evaluation
        Args:
            X (pd.Dataframe): input
            clustering_name (sklearn function): name of clustering algorithm
            label_x (str): name of x label
            label_y (str): name of y label
            label_z (str): name of z label
        Returns:
            -
    """
    # data
    X = dataframe[x_feature]
    y = dataframe[y_feature]
    
    # tf-idf
    tfidf = TfidfVectorizer(**params)
    X = (tfidf.fit_transform(X)).todense()
    values = tfidf.get_feature_names_out()

    # Split dataset into train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, 
                                                        y, 
                                                        stratify=y, 
                                                        test_size=0.3, 
                                                        random_state=1)

    # train model
    model.fit(X_train, y_train)
                 
    # results
    y_pred = model.predict(X_test)

    # performance
    print( "Accuracy: "+ str(accuracy_score(y_test, y_pred, normalize=True)*100) + " %")
    
    return model, X_test, y_test

                       
def find_optimal_clusters(X, clustering, max_clusters):
    """General function for model creation and evaluation
        Args:
            X (pd.Dataframe): input
            clustering (sklearn function): name of clustering algorithm
            max_clusters (int): number of maximum clusters
        Returns:
            -
    """
    # instantiate the clustering model 
    model = clustering
    
    # instantiate the visualizer
    plt.figure(figsize=(12,8))
    visualizer = KElbowVisualizer(model, k=(2, max_clusters))
    visualizer.fit(X)  
    visualizer.show()  
    
    return visualizer


class ClusteringText:
 
    
    def __init__(self, 
                  dataframe, 
                  feature, 
                  category_filter, 
                  clustering_type, 
                  clustering_params,
                  max_clusters, 
                  tfidf_params):

        """General function for clustering model creation, optimization and visualization

        Args:
            dataframe (pd.Dataframe): data source
            feature (String): column to apply TfidfVectorizer
            category_filter List(String): filter by category type in product_category
            clustering_type:, 
            max_clusters:, 
            tfidf_params (dict): parameters of TfidfVectorizer
        Returns:
            -
        """
        self.dataframe = dataframe.copy()
        self.feature = feature
        self.category_filter = category_filter
        self.clustering_type = clustering_type
        self.clustering_params = clustering_params
        self.max_clusters = max_clusters
        self.tfidf_params = tfidf_params
        self.vectorizer = ''
        self.text_vectorized = []
        self.text_vectorized_pca = []
        self.human_labels = []
        self.automatic_labels = []
        self.automatic_labels_pca = []
        
    
    def find_optimal_clusters_function(self, vectorizer):
        """"""
        self.vectorizer = vectorizer
        vocab = []

        #
        if self.category_filter:
            self.dataframe = self.dataframe[self.dataframe.product_category == self.category_filter]

        # Overview
        #info(self.dataframe)
        
        #
        X = self.dataframe[self.feature]

        #
        if self.vectorizer == "tfidf":
          
            # tf-idf
            tfidf = TfidfVectorizer(**self.tfidf_params)
            self.text_vectorized = tfidf.fit_transform(X)
            values = tfidf.get_feature_names_out()
            
            print("\nElbow method:\n")
            find_optimal_clusters(self.text_vectorized, 
                                  self.clustering_type(), 
                                  self.max_clusters)


        elif self.vectorizer == "gensim":
            
            # tag each paragraph
            corpus = []
            for j, elem in enumerate(X):
                corpus.append(TaggedDocument(elem, [j]))
         
            
            #
            d2v_model = Doc2Vec(corpus, 
                                #size = 100,
                                #window = 10,
                                min_count = 1,
                                #workers=7, dm = 1,
                                #alpha=0.025,
                                #min_alpha=0.001
                               )
            
            #
            #vocab = d2v_model.dv.key_to_index
            
            # appending all the vectors in a list for training
            for i in range(len(corpus)):
                self.text_vectorized.append(d2v_model.docvecs[i])
                
        return self.text_vectorized
        

    def text_clustering_function(self,  
                      n_comp_pca):
        """""" 
        pca = None
        
        #
        self.human_labels = self.dataframe.product_category.values
        clustering = self.clustering_type(**self.clustering_params)
        
        #print("text_vectorized[0]: " + str(self.text_vectorized[0].shape))
        
        # fit clustering
        start_at = datetime.now()
        clustering.fit(self.text_vectorized)
        end_at = datetime.now()
        self.automatic_labels = clustering.labels_
        self.dataframe['automatic_labels'] = self.automatic_labels

        # pca
        pca = decomposition.PCA(n_components=n_comp_pca)
        if str(type(self.text_vectorized)) == "<class 'scipy.sparse.csr.csr_matrix'>":
            self.text_vectorized = self.text_vectorized.todense()
            self.text_vectorized_pca = pca.fit_transform(self.text_vectorized)
        elif str(type(self.text_vectorized)) == "<class 'list'>":
            self.text_vectorized_pca = pca.fit_transform(self.text_vectorized)
       
        # re-fit clustering after pca
        start_at_pca = datetime.now()
        clustering.fit(self.text_vectorized_pca)
        end_at_pca = datetime.now()
        self.automatic_labels_pca = clustering.labels_ 
        self.dataframe['automatic_labels_pca'] = self.automatic_labels_pca

        # pca explained variance
        print("\n\nPCA:")
        display_scree_plot(pca, n_comp_pca)
                
        # t-sne 
        self.dataframe = tsne_data_compute_and_record_function(self.dataframe, self.text_vectorized, self.text_vectorized_pca)
        
        # plot
        plot_4_grid(self.dataframe, self.automatic_labels, self.automatic_labels_pca, self.human_labels, n_comp_pca)
            
        # stats
        stats = clustering_stats_function("\nclustering stats:\n", self.automatic_labels, self.human_labels, self.text_vectorized, start_at, end_at)
        stats_pca = clustering_stats_function("\nclustering stats after pca:\n", self.automatic_labels_pca, self.human_labels, self.text_vectorized_pca, start_at_pca, end_at_pca)

        
        return self.human_labels, self.automatic_labels, self.automatic_labels_pca, stats, stats_pca, self.text_vectorized_pca, self.dataframe
    
    
    def clustering_3d_function(self, labels, pca):
        """"""
        #
        if pca == True:
            x, y, z = "tsne-3d-one-pca", "tsne-3d-two-pca", "tsne-3d-three-pca"
      
        elif pca == False:
            x, y, z = "tsne-3d-one", "tsne-3d-two", "tsne-3d-three"
               
        #    
        if labels == 'automatic' and pca == True:
            labels = self.automatic_labels_pca
            
        elif labels == 'automatic' and pca == False:
            labels = self.automatic_labels
            
        elif labels == 'human':
            labels = self.human_labels
                        
        #
        plot_3d_function(self.dataframe, 
                         labels, 
                         x, 
                         y, 
                         z)
        
        return 
            

def get_thumbnail(folder, path):
    """xxx
        Args:
            folder: , 
            path:, 
          
        Returns:
            i:
    """
    path = folder + path 
    i = Image.open(path)    
    
    return i


def image_details_function(dataframe, image_feature):
    """xxx
        Args:
            dataframe(pd.Dataframe): input, 
            image_feature, 
          
        Returns:
            df_image:
    """
    #
    df_image = dataframe.copy()

    # new column for image thumbnail
    df_image['image_thumbnail'] = dataframe[image_feature].apply(lambda x: get_thumbnail('images/', x))
    # new column for image size
    df_image['image_array'] = df_image['image_thumbnail'].apply(lambda x: np.array(x))
    # new column for image size
    df_image['image_size'] = df_image['image_thumbnail'].apply(lambda x: np.array(x).shape)
    
    return df_image

#
def display_histogramm(matrix, bins, density, cumulative, xlabel, ylabel, title):
    """xxx
        Args:
            matrix, 
            bins, 
            density, 
            cumulative, 
            xlabel, 
            ylabel, 
            title
            
        Returns:
            -
    """
    # generate and display histogramm 
    fig, ax = plt.subplots(figsize=(15, 7))
    n, bins, patches = plt.hist(matrix.flatten(),
                                bins=range(bins),
                                density=density,
                                cumulative=cumulative)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    plt.show()


def display_both(image_1, cmap_1, title_1, image_2, cmap_2, title_2):
    """xxx
        Args:
            image_1:, 
            cmap_1:, 
            title_1:, 
            image_2:, 
            cmap_2:, 
            title_2:
            
        Returns:
            -
    """
    # Display traning image and testing image
    fx, plots = plt.subplots(1, 2, figsize=(20,10))
    plots[0].set_title(title_1)
    plots[0].imshow(image_1, cmap=cmap_1)
    plots[1].set_title(title_2)
    plots[1].imshow(image_2, cmap=cmap_2)
    plt.show()
    
    return


def image_pretreatment_function(image, color, length, height):
    """xxx
        Args:
            image (xxx): xxx
            color (str): image color, 
            length (int): image length, 
            height (int): image height
            
        Returns:
            image_array, 
            image
    """
    # optimize exposition
    image = ImageOps.autocontrast(image)
    
    # optimize contrast
    image = ImageOps.equalize(image)
    
    # convert color
    image_array = np.array(image)
    if color == 'COLOR_BGR2RGB':
        image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    elif color == 'COLOR_RGB2GRAY':
        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)   
        
    # apply gaussian noise to image     
    noise = np.random.normal(0, 7, image_array.shape)
    image = Image.fromarray(image_array + noise, 'RGB')
    
    # apply smoothing by moving average 
    image = image.filter(ImageFilter.BoxBlur(1))
    
    # resize
    image_array = np.array(image)
    image_array = cv2.resize(image_array, (length, height))
    image = Image.fromarray(image_array)
    
    return image_array, image
    

def image_preprocessing(dataframe, image_feature, color, length, height):
    """xxx
        Args:
            dataframe(pd.Dataframe): input
            image_feature (str): feature for images
            color (str): image color, 
            length (int): image length, 
            height (int): image height
            
        Returns:
            dataframe(pd.Dataframe): output
    """
  
    #
    dataframe['image_array_preprocessed'] = dataframe[image_feature].apply(lambda x: image_pretreatment_function(x, 
                                                                                                                 color, 
                                                                                                                 length, 
                                                                                                                 height)[0])
    
    #
    dataframe['image_array_preprocessed_size'] = dataframe['image_array_preprocessed'].apply(lambda x: x.shape)
    
    #
    dataframe['image_thumbnail_preprocessed'] = dataframe[image_feature].apply(lambda x: image_pretreatment_function(x, 
                                                                                                           color, 
                                                                                                           length, 
                                                                                                           height)[1])
    
    return dataframe



def keypoint_and_descriptor_function(dataframe, image_feature):
    """xxx
        Args:
            dataframe(pd.Dataframe): input
            image_feature (str): feature for images
        Returns:
            -
    """
    #sift_descriptors =  []
    sift = cv2.SIFT_create() #nfeatures=
    
    #
    dataframe['keypoints'] = dataframe[image_feature].apply(lambda x: sift.detectAndCompute(x, None)[0])
    #
    dataframe['keypoints_number'] = dataframe['keypoints'].apply(lambda x: len(x))
    #
    dataframe['descriptors'] = dataframe[image_feature].apply(lambda x: sift.detectAndCompute(x, None)[1])


    return dataframe


def sift_clustering_function(dataframe, 
                             x_features, 
                             y_features,
                             clustering,
                             clustering2,
                             clustering_params, 
                             clustering2_params,
                             n_comp_pca):
    """xxx
        Args:
            dataframe(pd.Dataframe): input
            x_features (str): , 
            y_feature (str):,
            clustering (sklearn function): name of clustering algorithm,
            clustering_params:, 
            n_comp_pca (int): number of PCA components
        Returns:
            -
    """
    # 1- features extraction
    # data
    dataframe = dataframe.dropna(how='any', axis=0)
    X_features = np.vstack(dataframe[x_features].tolist())
    
    y = dataframe[y_features]
    human_labels = y.values
 
    # 2- bag-of-features
    # clustering
    clustering = clustering(**clustering_params)
    clustering.fit(X_features)
    
    # 3- Histograms   
    k = np.size(dataframe.product_category.unique())*10
        
    '''def histo_func(des):
        
        histo = np.zeros(k)  
        
        for d in des:
            des = np.array(des)
            d = d.reshape(1, -1)
            idx = clustering.predict(d)
            histo[idx] += 1/len(des)
            
        print(histo)
        
        return histo
    
    dataframe['histogram'] = dataframe[[x_features]].apply(lambda x: histo_func(x))'''
    
    hist_list = []
    
    for x in dataframe[x_features]:
        histo = np.zeros(k)  
        
        for des in x:
            #des = np.array(des)
            des = des.reshape(1, -1)
            idx = clustering.predict(des)
            histo[idx] += 1/len(x)

        hist_list.append(histo) 
            
    # prepare data          
    X_features2 = hist_list
    #X_features2 = X_features2.tolist()  
    X_features2 = np.array([np.array(val) for val in X_features2]) 
    X_features2 = np.nan_to_num(X_features2)
    
    # clustering
    clustering2 = clustering2(**clustering2_params)
    start_at = datetime.now()
    clustering2.fit(X_features2)
    end_at = datetime.now()
    automatic_labels = clustering2.labels_ 
    dataframe['automatic_labels'] = automatic_labels
           
    # pca
    pca = decomposition.PCA(n_components=n_comp_pca)
    X_features2_pca = pca.fit_transform(X_features2)
    
    # re-fit clustering after pca
    start_at_pca = datetime.now()
    clustering2.fit(X_features2_pca)
    end_at_pca = datetime.now()
    automatic_labels_pca = clustering2.labels_    
    dataframe['automatic_labels_pca'] = automatic_labels_pca
    
    # pca explained variance
    print("\n\nPCA:")
    display_scree_plot(pca, n_comp_pca)

    # t-sne
    dataframe = tsne_data_compute_and_record_function(dataframe, X_features2, X_features2_pca)
         
    # plot
    plot_4_grid(dataframe, automatic_labels, automatic_labels_pca, human_labels, n_comp_pca)

    # stats
    stats = clustering_stats_function("\nclustering stats:\n", automatic_labels, human_labels, X_features2, start_at, end_at)
    stats_pca = clustering_stats_function("\nclustering stats after pca:\n", automatic_labels_pca, human_labels, X_features2_pca, start_at_pca, end_at_pca)

    # output
    labels = [human_labels, automatic_labels, automatic_labels_pca]
    
    return labels, dataframe, stats, stats_pca, X_features2, X_features2_pca
 
    
def my_vgg16_implementation_function():
    """xxx
        Args:
            -
       
        Returns:
            -
    """
    # create void neural network
    my_VGG16 = Sequential()  

    ### 2 convolution layers + polling layer (x2) ###
    my_VGG16.add(Conv2D(64, (3, 3), input_shape=(224, 224, 3), padding='same', activation='relu'))
    my_VGG16.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    my_VGG16.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

    my_VGG16.add(Conv2D(128, (3, 3), input_shape=(112, 112, 128), padding='same', activation='relu'))
    my_VGG16.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    my_VGG16.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

    ### 3 convolution layers + polling layer (x3) ###
    my_VGG16.add(Conv2D(256, (3, 3), input_shape=(56, 56, 256), padding='same', activation='relu'))
    my_VGG16.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    my_VGG16.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    my_VGG16.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

    # add first layer of convolution followed by ReLU layer
    my_VGG16.add(Conv2D(512, (3, 3), input_shape=(28, 28, 512), padding='same', activation='relu'))
    my_VGG16.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
    my_VGG16.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
    my_VGG16.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

    # add first layer of convolution followed by ReLU layer
    my_VGG16.add(Conv2D(512, (3, 3), input_shape=(14, 14, 512), padding='same', activation='relu'))
    my_VGG16.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
    my_VGG16.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
    my_VGG16.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))


    ### 3 fully connected layer ###
    # conversion of 3D matrix into 1D array
    my_VGG16.add(Flatten())
    my_VGG16.add(Dense(4096, activation='relu'))
    my_VGG16.add(Dense(4096, activation='relu'))
    my_VGG16.add(Dense(1000, activation='softmax'))

    my_VGG16.summary()
    
    
    
def pre_trained_model_as_classifier_function(image_path, model, x_size, y_size):
    """xxx
        Args:
            image_path, 
            model, 
            x_size, 
            y_size
    
        Returns:
            -
    """
    # summarize the model
    model.summary()

    # load an image from file
    image = load_img(image_path, target_size=(x_size, y_size))
    
    # convert the image pixels to a numpy array
    image = img_to_array(image)

    # reshape data for the model
    image = image.reshape((1, 
                           image.shape[0], 
                           image.shape[1], 
                           image.shape[2]))

    # prepare the image for the VGG model
    image = preprocess_input(image)
    
    # predict the probability across all output classes
    y = model.predict(image)
    
    # convert the probabilities to class labels
    label = decode_predictions(y)

    # retrieve the most likely result, e.g. highest probability
    label_0 = label[0][0]

    # print the classification
    print('\nClassification: ', image_path)
    print('%s (%.2f%%)\n' % (label_0[1], label_0[2]*100))
    print('Top 3:\n', 
          (decode_predictions(y, top=3)[0][0][1]),  
          '(%.2f%%)' % (decode_predictions(y, top=3)[0][0][2]*100), 
          '\n', 
          (decode_predictions(y, top=3)[0][1][1]), 
          '(%.2f%%)' % (decode_predictions(y, top=3)[0][1][2]*100), 
           '\n', 
          (decode_predictions(y, top=3)[0][2][1]), 
          '(%.2f%%)' % (decode_predictions(y, top=3)[0][2][2]*100))
    
    
    
def my_custom_vgg_for_classification_function():
    """xxx
        Args:
            -
    
        Returns:
            -
    """
    # load pre trained VGG-16 on ImageNet without fully-connected layers
    conv_base = VGG16(weights="imagenet", 
                  include_top=False, 
                  input_shape=(224, 224, 3))

    # 2 : features extraction
    for layer in conv_base.layers:
        layer.trainable = False

    # retrieve output layer of the network 
    top_model = conv_base.output

    # add new classifier layers
    top_model = Flatten(name="flatten")(top_model)
    top_model_add = Dense(4096, activation='relu')(top_model)
    top_model_add = Dense(4096, activation='relu')(top_model_add)
    predictions = Dense(7, activation='softmax')(top_model_add)

    # define new model
    my_model = Model(inputs=conv_base.input, outputs=predictions)

    # compile model 
    my_model.compile(loss="sparse_categorical_crossentropy", 
                      optimizer=SGD(lr=0.0001, momentum=0.9), 
                      metrics=["accuracy"])

    # summarize the model
    my_model.summary()
    
    return my_model  
    
    
def my_custom_vgg_classification_function(dataframe, 
                                          model, 
                                          x_feature, 
                                          y_feature, 
                                          epochs, 
                                          batch_size):
    """xxx
        Args:
            dataframe (pd.Dataframe): input
            model, 
            x_feature, 
            y_feature, 
            epochs, 
            batch_size
        
        Returns:
            y_test, 
            y_pred
    """
    # data
    X = dataframe[x_feature]
    y = dataframe[y_feature]
    
    #
    le = LabelEncoder()
    y = le.fit_transform(y)

    #
    X = X.apply(lambda x: img_to_array(x))
    
    #
    X = X.apply(lambda x: preprocess_input(x))
    
    #
    X = np.array([np.array(val) for val in X]) 

    # Split dataset into train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, 
                                                        y, 
                                                        test_size=0.2, 
                                                        random_state=1234,
                                                        shuffle=True,
                                                        stratify=y)
    
    #print("X_test0:" + str(X_test[0].shape))
    
    
    print("Custom VGG fitting...")
    # train model
    model.fit(X_train, 
                 y_train, 
                 epochs=epochs, 
                 batch_size=batch_size, 
                 verbose=2)
    
  
    print("predicting...")
    # results
    y_pred = model.predict(X_test)
    
    #
    y_pred = np.argmax(y_pred, axis=1)
    
    #
    y_pred = le.inverse_transform(y_pred)
    y_test = le.inverse_transform(y_test)

    
    # performance
    print("VGG Model Accuracy without Fine-Tuning: " + str(accuracy_score(y_test, y_pred, normalize=True)*100) + " %")
    
    
    return y_test, y_pred


def vgg_for_clustering_function():
    """xxx
        Args:
            -
        Returns:
            -
    """
    # load pre trained VGG-16 on ImageNet without fully-connected layers
    conv_base = VGG16(weights="imagenet", 
                  include_top=False, 
                  input_shape=(224, 224, 3))

    # 2 : features extraction
    for layer in conv_base.layers:
        layer.trainable = False

    # define new model
    my_model = Model(inputs=conv_base.input, outputs=conv_base.output)

    # compile model 
    my_model.compile(loss="sparse_categorical_crossentropy", 
                      optimizer=SGD(lr=0.0001, momentum=0.9), 
                      metrics=["accuracy"])
    #
    my_model.summary()
    
    return my_model


def vgg_clustering_function(dataframe, 
                            vgg, 
                            x_feature, 
                            y_feature, 
                            clustering,
                            clustering_params, 
                            n_comp_pca):
    """xxx
        Args:
            dataframe (pd.Dataframe): input
            vgg (keras.applications.vgg16): VGG-16 model
            x_feature (str): feature for images, 
            y_feature (str): feature for labels, 
            clustering (sklearn function): name of clustering algorithm,
            clustering_params:, 
            n_comp_pca (int): number of PCA components

        Returns:
            labels list(list(int/str)): , 
            dataframe (pd.Dataframe): output
    """
    # data
    X = dataframe[x_feature]
    y = dataframe[y_feature]
    
    #
    human_labels = y.values
    
    #
    X = X.apply(lambda x: img_to_array(x))
    
    #
    X = X.apply(lambda x: preprocess_input(x))
    
    #
    X = np.array([np.array(val) for val in X]) 
        
    #    
    X_features = vgg.predict(X)
    X_features = np.array([np.array(val) for val in X_features]) 
    X_features = np.array([val.flatten() for val in X_features]) 
    
    # train model
    clustering = clustering(**clustering_params)
    start_at = datetime.now()
    clustering.fit(X_features)
    end_at = datetime.now()
    automatic_labels = clustering.labels_
    dataframe['automatic_labels'] = automatic_labels
    
    # pca
    pca = decomposition.PCA(n_components=n_comp_pca)
    X_features_pca = pca.fit_transform(X_features)
    
    # re-fit clustering after pca
    start_at_pca = datetime.now()
    clustering.fit(X_features_pca)
    end_at_pca = datetime.now()
    automatic_labels_pca = clustering.labels_
    dataframe['automatic_labels_pca'] = automatic_labels_pca
    
    # pca explained variance
    print("\n\nPCA:")
    display_scree_plot(pca, n_comp_pca)

    # t-sne
    dataframe = tsne_data_compute_and_record_function(dataframe, X_features, X_features_pca)
      
    # plot
    plot_4_grid(dataframe, automatic_labels, automatic_labels_pca, human_labels, n_comp_pca)

    # clustering stats
    stats = clustering_stats_function("\nclustering stats:\n", automatic_labels, human_labels, X_features, start_at, end_at)
    stats_pca = clustering_stats_function("\nclustering stats after pca:\n", automatic_labels_pca, human_labels, X_features_pca, start_at_pca, end_at_pca)

    labels = [human_labels, automatic_labels, automatic_labels_pca]

    return labels, dataframe, stats, stats_pca, X_features, X_features_pca



def plot_confusion_matrix_function(y_true, y_pred, title):
    """ xxx
    Args:
        y_true list(str):
        y_pred list(int):
        title (str): 
    Returns:
        -
    """
    # Create a DataFrame with labels and varieties as columns: df
    df = pd.DataFrame({'Labels': y_true, 'Clusters': y_pred})

    # Create crosstab: ct
    ct = pd.crosstab(df['Labels'], df['Clusters'])

    # plot the heatmap for correlation matrix
    fig, ax = plt.subplots(figsize=(12, 12))

    sns.heatmap(ct.T, 
                 square=True, 
                 annot=True, 
                 annot_kws={"size": 17},
                 fmt='.2f',
                 cmap='Blues',
                 cbar=False,
                 ax=ax)

    ax.set_title(title, fontsize=16)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=17)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, ha="right", fontsize=17)
    ax.set_ylabel("clusters", fontsize=17)
    ax.set_xlabel("labels", fontsize=17)

    plt.show()



def find_optimal_clusters_function(self, vectorizer):
    """ xxx
    Args:
        vectorizer :
    Returns:
        -
    """
    #
    if category_filter:
        dataframe = dataframe[dataframe.product_category == category_filter]

    # Overview
    info(dataframe)

    #
    X = dataframe[feature]

    # tf-idf
    tfidf = TfidfVectorizer(**tfidf_params)
    text_vectorized = tfidf.fit_transform(X)
    values = tfidf.get_feature_names_out()

    find_optimal_clusters(text_vectorized, 
                          clustering_type(), 
                          max_clusters)
    
    
    
def clustering_3d_function(dataframe, human_labels, automatic_labels, automatic_labels_pca, labels_type, pca):
    """ xxx
    Args:
        dataframe (pd.Dataframe): input, 
        human_labels: list(str): original labels (product_category), 
        automatic_labels list(int): labels generated by the clustering, 
        automatic_labels_pca list(int): labels generated by the clustering after PCA, 
        labels_type (str): original(human) or computed(automatic), 
        pca (Boolean): display data with or without PCA 
    Returns:
        -
    """
    #
    if pca == True:
        x, y, z = "tsne-3d-one-pca", "tsne-3d-two-pca", "tsne-3d-three-pca"

    elif pca == False:
        x, y, z = "tsne-3d-one", "tsne-3d-two", "tsne-3d-three"

    #    
    if labels_type == 'automatic' and pca == True:
        labels = automatic_labels_pca

    elif labels_type == 'automatic' and pca == False:
        labels = automatic_labels

    elif labels_type == 'human':
        labels = human_labels

    #
    plot_3d_function(dataframe, 
                     labels, 
                     x, 
                     y, 
                     z)

    return 
    
    
def text_image_clustering_function(dataframe, 
                                   text_array_feature, 
                                   image_array_feature,
                                   y_feature, 
                                   clustering,
                                   clustering_params, 
                                   n_comp_pca):
    """General function for model creation and evaluation
        Args:
            dataframe (pd.Dataframe): input
            text_array_feature, 
            image_array_feature,
            y_feature (str): , 
            clustering (sklearn function): name of clustering algorithm,
            clustering_params, 
            n_comp_pca (int): number of PCA components
            
        Returns:
            labels, 
            data_vectorized
    """
    # normalize data between 0 and 1
    X1 = dataframe[text_array_feature] 
    X1 = X1.apply(lambda x : np.asarray(x))
    X1 = X1.apply(lambda x : (x - np.min(x))/np.ptp(x))
    
    X2 = dataframe[image_array_feature] 
    X2 = X2.apply(lambda x : np.asarray(x))
    X2 = X2.apply(lambda x : (x - np.min(x))/np.ptp(x))
    
    # concatenate text and image data
    X = X1.apply(lambda x: x.tolist()) + X2.apply(lambda x: x.tolist())
    
    # convert data into numpy matrix
    X_features = np.array([np.array(val) for val in X]) 
    data_vectorized = np.copy(X_features)
    
    # labels
    human_labels = dataframe[y_feature].values
    
    # train model
    clustering = clustering(**clustering_params)
    start_at = datetime.now()
    clustering.fit(X_features)
    end_at = datetime.now()
    automatic_labels = clustering.labels_
    dataframe['automatic_labels'] = automatic_labels
    
    # pca
    pca = decomposition.PCA(n_components=n_comp_pca)
    X_features_pca = pca.fit_transform(X_features)
    
    # re-fit clustering after pca
    start_at_pca = datetime.now()
    clustering.fit(X_features_pca)
    end_at_pca = datetime.now()
    automatic_labels_pca = clustering.labels_   
    dataframe['automatic_labels_pca'] = automatic_labels_pca
    
    # pca explained variance
    print("\n\nPCA variance:\n")
    display_scree_plot(pca, n_comp_pca)

    # t-sne 
    dataframe = tsne_data_compute_and_record_function(dataframe, X_features, X_features_pca)
         
    # plot
    plot_4_grid(dataframe, automatic_labels, automatic_labels_pca, human_labels, n_comp_pca)

    # stats
    stats = clustering_stats_function("\nclustering stats:\n", automatic_labels, human_labels, X_features, start_at, end_at)
    stats_pca = clustering_stats_function("\nclustering stats after pca:\n", automatic_labels_pca, human_labels, X_features_pca, start_at_pca, end_at_pca)
    
    # Output
    labels = [human_labels, automatic_labels, automatic_labels_pca]
    
    return labels, data_vectorized, stats, stats_pca, dataframe


def tsne_data_compute_and_record_function(dataframe, X_features, X_features_pca):                                      
    """ xxx
    Args:
        dataframe (pd.Dataframe): input, 
        X_features, 
        X_features_pca
    Returns:
        dataframe (pd.Dataframe): output
    """
    # t-sne 2d
    tsne = TSNE(n_components=2, random_state=0)
    tsne_results_2d = tsne.fit_transform(X_features) 
    tsne_results_2d_pca = tsne.fit_transform(X_features_pca) 
    
    # t-sne 3d
    tsne = TSNE(n_components=3, random_state=0)
    tsne_results_3d = tsne.fit_transform(X_features) 
    tsne_results_3d_pca = tsne.fit_transform(X_features_pca) 
    
    # data 2d
    dataframe['tsne-2d-one'] = tsne_results_2d[:,0]
    dataframe['tsne-2d-two'] = tsne_results_2d[:,1]
    dataframe['tsne-2d-one-pca'] = tsne_results_2d_pca[:,0]
    dataframe['tsne-2d-two-pca'] = tsne_results_2d_pca[:,1]

    # data 3d
    dataframe['tsne-3d-one'] = tsne_results_3d[:,0]
    dataframe['tsne-3d-two'] = tsne_results_3d[:,1]
    dataframe['tsne-3d-three'] = tsne_results_3d[:,2]
    dataframe['tsne-3d-one-pca'] = tsne_results_3d_pca[:,0]
    dataframe['tsne-3d-two-pca'] = tsne_results_3d_pca[:,1]
    dataframe['tsne-3d-three-pca'] = tsne_results_3d_pca[:,2]
    
    return dataframe


def clustering_stats_function(title, 
                              automatic_labels, 
                              human_labels,
                              X_features,
                              start_at,
                              end_at):
    """ xxx
    Args:
        title (str): , 
        automatic_label:, 
        human_labels:,
        X_features:,
        start_at:,
        end_at:
    Returns:
        -
    """
    
    ari = adjusted_rand_score(automatic_labels, human_labels)
    silhouette = silhouette_score(X_features, automatic_labels, metric="sqeuclidean")
    running_time = (end_at - start_at).total_seconds()
    
    print(title)
    print("ARI: " + str(ari))
    print("Silhouette: " + str(silhouette))
    print("Clustering fit time: " + str(running_time))
    
    stats = [ari, silhouette, running_time]
    
    return stats


def display_scree_plot(pca, n_comp_pca):
    """ Plots PCA variance histogram
    Args:
        pca (sklearn.decomposition._pca.PCA):
        n_comp_pca (int): number of components
    Returns:
        -
    """
    print("\n(PCA) explained variance for " + str(n_comp_pca) + " components: {}\n".format(pca.explained_variance_ratio_.cumsum()[-1]))
    
    scree = pca.explained_variance_ratio_*100
    plt.figure(figsize=(7,6))
    plt.bar(np.arange(len(scree))+1, scree)
    plt.plot(np.arange(len(scree))+1, scree.cumsum(),c="red",marker='o')
    plt.xlabel("Rang de l'axe d'inertie")
    plt.ylabel("Pourcentage d'inertie")
    plt.title("Eboulis des valeurs propres")
    plt.show(block=False)
    
    
def plot_4_grid(dataframe, 
                automatic_labels, 
                automatic_labels_pca, 
                human_labels, 
                n_comp_pca):
    """ Plots 4 scatter plots with labels
    Args:
        dataframe (pd.Dataframe): input, 
        automatic_labels list(int): labels generated by the clustering, 
        automatic_labels_pca list(int): labels generated by the clustering after PCA, 
        human_labels: list(str): original labels (product_category), 
        n_comp_pca (int): number of PCA components

    Returns:
        -
    """
    print("\n\nClustering:\n")
    
    # plot
    fig, axes = plt.subplots(2, 2, figsize=(18, 16))

    sns.scatterplot(
            ax=axes[0][0],
            x="tsne-2d-one", y="tsne-2d-two",
            hue=automatic_labels,
            palette=sns.color_palette("hls", len(np.unique(automatic_labels))),
            data=dataframe,
            legend="brief",
            alpha=1)

    # Put the legend out of the figure
    axes[0][0].legend(loc='best', bbox_to_anchor=(1, 0.5))
    axes[0][0].set_title("Automatic labelling")

    sns.scatterplot(
            ax=axes[0][1],
            x="tsne-2d-one", y="tsne-2d-two",
            hue=human_labels,
            palette=sns.color_palette("hls", len(np.unique(human_labels))),
            data=dataframe,
            legend="brief",
            alpha=1)

    # Put the legend out of the figure
    axes[0][1].legend(loc='best', bbox_to_anchor=(1, 0.5))
    axes[0][1].set_title("Human labelling")

    sns.scatterplot(
            ax=axes[1][0],
            x="tsne-2d-one-pca", y="tsne-2d-two-pca",
            hue=automatic_labels,
            palette=sns.color_palette("hls", len(np.unique(automatic_labels_pca))),
            data=dataframe,
            legend="brief",
            alpha=1)

    # Put the legend out of the figure
    axes[1][0].legend(loc='best', bbox_to_anchor=(1, 0.5))
    axes[1][0].set_title('Automatic labelling (PCA n_comp=' + str(n_comp_pca) + ')')

    sns.scatterplot(
            ax=axes[1][1],
            x="tsne-2d-one-pca", y="tsne-2d-two-pca",
            hue=human_labels,
            palette=sns.color_palette("hls", len(np.unique(human_labels))),
            data=dataframe,
            legend="brief",
            alpha=1)

    # Put the legend out of the figure
    axes[1][1].legend(loc='best', bbox_to_anchor=(1, 0.5))
    axes[1][1].set_title('Human labelling (PCA n_comp=' + str(n_comp_pca) + ')')

    plt.show()
    

def plot_3d_function(X, labels, label_x, label_y, label_z):
    """Scatter plot in 3d 
        Args:
            X (pd.Dataframe): input
            label_x (str): name of x label
            label_y (str): name of y label
            label_z (str): name of z label
        Returns:
            -
    """
    #
    fig = px.scatter_3d(X, 
                        x=label_x,
                        y=label_y,
                        z=label_z,
                        title='',
                        labels={'0': label_x, '1': label_y, '2': label_z},
                        color=labels
                       )
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    fig.show()
    
    
    
def image_sample_display(dataframe, n_sample, n_labels, feature):
    """xxx 
        Args:
            dataframe, 
            n_sample, 
            n_labels,
            feature
        Returns:
            -
    """
    feature_elements_unique_list = dataframe[feature].unique()


    #
    def image_sample_by_feature(i):
        """"""

        df_sample = dataframe[dataframe[feature] == feature_elements_unique_list[i]].sample()
        product_label = df_sample.automatic_labels.tolist()[0]
        product_name = df_sample.product_name.tolist()[0]
        product_category = df_sample.product_category.tolist()[0]
        image_array = df_sample.image_array.tolist()[0]
        image_array = cv2.resize(image_array, (224, 224))

        return image_array, product_category, product_name, product_label


    # main
    image_list = []
    rows = n_labels
    columns = n_sample

    for i in range(rows):
        for j in range(columns):
            image_list.append(image_sample_by_feature(i))


    fig = plt.figure(figsize=(42, 32))

    for n in range(columns*rows):
        fig.add_subplot(rows, columns, n+1)
        plt.title("product name: " + str(image_list[n][2]) + "\nproduct category: " + str(image_list[n][1]) + "\nclustering label: " + str(image_list[n][3]))
        plt.imshow(image_list[n][0])
        plt.tick_params(axis='both',         
                        which='both',      
                        bottom=False,      
                        top=False,       
                        labelbottom=False)
        
    plt.show()
       
        

def sum_up_table_function(metric, order, **kwargs):
    """ 
    Stats sum up for the given algorithms 
    Args:
        metric (string): sklearn metric
        order (string): ascending or descending
    Returns:
        dataframe (pd.Dataframe): data output
    """                          
    # data
    d = {}
    
    for kwarg in kwargs:
        newline = {kwarg: kwargs[kwarg]}
        d.update(newline)

    index = ['ARI',
             'Silhouette',
             'Running time']

    # dataframe
    df = pd.DataFrame(data=d, index=index).transpose()

    order = order.lower()
    
    #
    if order == 'ascending':
        order = True 
    else: 
        order = False 
    
    #
    return df.sort_values(by=metric, axis=0, ascending=order, inplace=False, kind='quicksort')