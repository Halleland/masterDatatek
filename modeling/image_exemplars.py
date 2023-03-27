# based on concept: https://github.com/MaartenGr/Concept/tree/c0ab7b0bad4a2fb4003c33d0031f75ab342ef7b9
import concept
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import hdbscan
from hdbscan.flat import _new_select_clusters

def _extract_exemplars(clusterer, labels, n_clusters_in_use=False):
    image_nr = [i for i in range(len(labels))]

    # prep condensed tree
    condensed_tree = clusterer.condensed_tree_
    raw_tree = condensed_tree._raw_tree
    if n_clusters_in_use:
        clusters = _new_select_clusters(condensed_tree, clusterer.cluster_selection_epsilon)
        clusters = sorted(clusters)
    else:
        clusters = sorted(condensed_tree._select_clusters())
    cluster_tree = raw_tree[raw_tree['child_size']>1]

    # find points with maximum lambda in each leaf
    representative_images = {}
    cluster_labels = sorted(list(set(labels)))
    for cluster in cluster_labels:
        if cluster != -1:
            leaves = hdbscan.plots._recurse_leaf_dfs(cluster_tree, clusters[cluster])
            exemplars = np.array([])

            for leaf in leaves:
                max_lambda = raw_tree['lambda_val'][raw_tree['parent']==leaf].max()
                points = raw_tree['child'][(raw_tree['parent'] == leaf) &
                                            (raw_tree['lambda_val'] == max_lambda)]
                exemplars = np.hstack((exemplars, points))

            representative_images[cluster] = {"Indices": [int(index) for index in exemplars],
                                                "Names": [image_nr[int(index)] for index in exemplars]}

    return representative_images

def _extract_cluster_embeddings(image_embeddings,
                                    representative_images,
                                    labels):
        """ Create a concept cluster embedding for each concept cluster by
        averaging the exemplar embeddings for each concept cluster.
        Arguments:
            image_embeddings: All image embeddings
            representative_images: The representative images per concept cluster
        Updates:
            cluster_embeddings: The embeddings for each concept cluster
        Returns:
            exemplar_embeddings: The embeddings for each exemplar image
        """
        image_embeddings = np.array(image_embeddings)
        exemplar_embeddings = {}
        cluster_embeddings = []
        cluster_labels = sorted(list(set(labels)))
        for label in cluster_labels[1:]:
            embeddings = image_embeddings[np.array([index for index in
                                                    representative_images[label]["Indices"]])]
            cluster_embedding = np.mean(embeddings, axis=0).reshape(1, -1)

            exemplar_embeddings[label] = embeddings
            cluster_embeddings.append(cluster_embedding)

        cluster_embeddings = cluster_embeddings

        return exemplar_embeddings, cluster_embeddings
    

def _extract_exemplar_subset(exemplar_embeddings,
                            representative_images,
                            cluster_embeddings,
                            labels, diversity=0.3):
        """ Use MMR to filter out images in the exemplar set
        Arguments:
            exemplar_embeddings: The embeddings for each exemplar image
            representative_images: The representative images per concept cluster
        Returns:
            selected_exemplars: A selection (8) of exemplar images for each concept cluster
        """
        cluster_labels = sorted(list(set(labels)))
        selected_exemplars = {cluster: concept._mmr.mmr(cluster_embeddings[cluster],
                                           exemplar_embeddings[cluster],
                                           representative_images[cluster]["Indices"],
                                           diversity=diversity,
                                           top_n=9)
                              for index, cluster in enumerate(cluster_labels[1:])}

        return selected_exemplars

def _cluster_representation(images,
                            selected_exemplars,
                            labels):
        """ Cluster exemplars into a single image per concept cluster
        Arguments:
            images: A list of paths to each image
            selected_exemplars: A selection of exemplar images for each concept cluster
        """
        # Find indices of exemplars per cluster
        cluster_labels = sorted(list(set(labels)))
        sliced_exemplars = {cluster: [[j for j in selected_exemplars[cluster][i:i + 3]]
                                      for i in range(0, len(selected_exemplars[cluster]), 3)]
                            for cluster in cluster_labels[1:]}
        
        # combine exemplars into a single image
        cluster_images = {}
        for cluster in cluster_labels[1:]:
            images_to_cluster = [[Image.open(images[index]) for index in sub_indices] for sub_indices in sliced_exemplars[cluster]]
            cluster_image = concept._visualization.get_concat_tile_resize(images_to_cluster)
            cluster_images[cluster] = cluster_image

            # Make sure to properly close images
            '''
            for image_list in images_to_cluster:
                for image in image_list:
                    image.close()
            '''

        return cluster_images


def find_image_represenation_of_clusters(clusterer:hdbscan.HDBSCAN, labels, image_embeddings, image_list, n_clusters_in_use=False):
    representative_images = _extract_exemplars(clusterer, labels, n_clusters_in_use)
    exemplar_embeddings, cluster_embeddings = _extract_cluster_embeddings(image_embeddings,
                                                        representative_images, labels)
    selected_exemplars = _extract_exemplar_subset(exemplar_embeddings,
                                                    representative_images,
                                                    cluster_embeddings, labels)
    cluster_images = _cluster_representation(image_list, selected_exemplars, labels)
    return cluster_images

def find_text_representation_of_clusters(clusterer, labels, n_clusters_in_use=False):
    exemplars = _extract_exemplars(clusterer, labels, n_clusters_in_use)
    pass