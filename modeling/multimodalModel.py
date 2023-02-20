import multimodal
import c_tf_idf
import image_exemplars

import umap
import hdbscan
import pandas as pd

# Plotting and visuals
import ipyplot
import seaborn as sns
import plotly.express as px

class MultimodalModel:
    def __init__(self, path_to_data, embedding_model, umap_model = None,
                clusterer = None, num_terms = 10,
                
                precomputed_text_embeds = None, precomputed_image_embeds =None,
                compute_embeddings=False):
        self.df = pd.read_table(path_to_data)
        self.texts, self.images = multimodal.get_image_and_text_from_file(path_to_data)
        self.embedding_model = embedding_model
        self.text_embeds = precomputed_text_embeds
        self.image_embeds = precomputed_image_embeds
        self.combined_embed = None
        if self.text_embeds is not None and self.image_embeds is not None:
            self.combine_embeds()

        if compute_embeddings:
            self.compute_embeddings()
        
        self.umap_model = umap_model
        if self.umap_model is None:
            self.set_default_umap()
        
        self.clusterer = clusterer
        if self.clusterer is None:
            self.set_default_clusterer()

        self.umap_embedding = None


        # For info
        self.tf_idf = None
        self.classes = None
        self.num_terms = num_terms
        self.cluster_images = None
        self.top_terms = None
    
    def set_default_umap(self):
        self.umap_model = umap.UMAP(n_neighbors=5,
                        n_components=5,
                        min_dist=0.1,
                        metric='cosine')

    def set_default_clusterer(self):
        self.clusterer = hdbscan.HDBSCAN(
    min_cluster_size=30)
    
    def combine_embeds(self):
        self.combined_embed = [a+b for a,b in zip(self.text_embeds, self.image_embeds)]

    def compute_embeddings(self):
        self.text_embeds = multimodal.get_embeddings_from_text(self.texts, self.embedding_model)
        self.image_embeds = multimodal.get_embeddings_from_images(self.images, self.embedding_model)
        self.combine_embeds()
    
    def fit(self):
        self.umap_embedding = self.umap_model.fit_transform(self.combined_embed)
        self.clusterer.fit(self.umap_embedding)

    def compute_c_tf_idf(self):
        """
        Requires fit umap
        """
        self.tf_idf, self.classes = c_tf_idf.find_c_tf_idf_from_hdbscan_model(self.df, self.clusterer)
        self.top_terms = c_tf_idf.get_top_terms(self.tf_idf, self.classes, self.num_terms)
    
    def fit_transform(self):
        if self.combined_embed is None:
            self.compute_embeddings()
        self.fit()
        self.compute_c_tf_idf()
        return self.clusterer.labels_
    
    def get_topic(self, i):
        return self.top_terms[i]

    def find_image_represenatives(self):
        """
        Requires trained clusterer
        """
        self.cluster_images = image_exemplars.find_image_represenation_of_clusters(self.clusterer, self.image_embeds, self.images)

    def plot_images(self):
        """
        Requires image represenatives
        """
        ipyplot.plot_images([*self.cluster_images.values()],custom_texts=self.top_terms, max_images=20, img_width=150)
    
    def plot_embedding(self):
        sns.scatterplot(x=self.umap_embedding[:,0], y=self.umap_embedding[:,1])
    
    def plot_clusters(self):
        self.clusterer.condensed_tree_.plot(select_clusters=True)
                
        palette = ['#1c17ff', '#faff00', '#8cf1ff', '#738FAB', '#030080', '#738fab']

        u = self.umap_embedding
        colors = [str(x) for x in self.clusterer.labels_]

        fig = px.scatter_3d(
            x=u[:,0], y=u[:,1], z=u[:,2],
            color=colors,
            color_discrete_sequence=palette
        )
        fig.update_traces(

        )
        return fig