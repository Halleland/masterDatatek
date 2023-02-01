from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from sklearn.preprocessing import minmax_scale
import base64
from io import BytesIO
from PIL import Image
import concept
from tqdm.auto import tqdm



def get_image_and_text_from_file(path):
    df = pd.read_table(path)
    images = df['b64_bytes'].apply(lambda x: BytesIO(base64.b64decode(x)))

    image_list = images.to_list()

    texts = df['context_page_description'].to_list()

    return texts, image_list

def get_embeddings_from_text(texts, embedding_model, batch_size=128):
    text_embeds = []
    n = len(texts)
    for i in tqdm(range(0, n, batch_size)):
        i_end = min(i+batch_size, n)
        batch = texts[i:i_end]
        batch_embed = embedding_model.encode(batch)
        text_embeds.extend(batch_embed.tolist())
    return text_embeds

def get_embeddings_from_images(image_list, embedding_model, batch_size=128):
    n = len(image_list)
    image_embeds = []
    for i in tqdm(range(0, n, batch_size)):
        i_end = min(i+batch_size, n)
        batch = image_list[i:i_end]
        images = [Image.open(img) for img in batch]
        batch_embed = embedding_model.encode(images)
        image_embeds.extend(batch_embed.tolist())
    return image_embeds

def get_combined_embeddings_from_img_and_text(texts, image_list, embedding_model, batch_size=128):
    text_embeds = get_embeddings_from_text(texts, embedding_model, batch_size)
    image_embeds = get_embeddings_from_images(image_list, embedding_model, batch_size)
    return [a+b for a,b in zip(text_embeds, image_embeds)]

