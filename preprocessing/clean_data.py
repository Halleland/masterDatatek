import pandas as pd

def extract_main_im_en(filepath):
    data = pd.read_table(filepath)
    data = data[data['language']=='en']
    main_image_en_data = data[data['is_main_image']==True]
    main_image_en_data.drop_duplicates('page_title')
    return main_image_en_data

def extract_main_im_en_to_file(filepath, new_name_without_filetype):
    data = extract_main_im_en(filepath)
    data.to_csv(new_name_without_filetype+'.tsv', sep='\t')