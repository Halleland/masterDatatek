import pandas as pd
import json

def extract_main_im_en(filepath):
    json_content = []
    with open(filepath, 'rb') as f:
        for line in f:
            if line:
                obj = json.loads(line)
                content = {}
                content['context_page_description']=""
                for element in obj['wit_features']:
                    language = None
                    main_image = False
                    if element.get('language'):
                        language = element['language']
                    if element.get('is_main_image'):
                        main_image = element['is_main_image']=='true'
                    
                    find_text_cond = element.get('context_page_description') and language=='en' and main_image

                    if find_text_cond:
                        content['context_page_description']= element['context_page_description']

                content['b64_bytes'] = obj['b64_bytes']

                # Check if captions or images are empty
                if content['context_page_description']!="" and content['b64_bytes'] != "":
                    json_content.append(content)

    return json_content

def extract_main_im_en_to_file(filepath, new_name_without_filetype):
    json_content = extract_main_im_en(filepath)
    df = pd.DataFrame.from_dict(json_content)
    df.to_csv(new_name_without_filetype+'.tsv', sep='\t', index=False)

def extract_multiple_files(raw_folder, processed_folder, filenames):
    for name in filenames:
        new_name = name.split('.')[0]
        filepath = f'{raw_folder}\\{name}'
        new_path = f'{processed_folder}\\processed_{new_name}'
        extract_main_im_en_to_file(filepath=filepath, new_name_without_filetype=new_path)

def load_multiple_files_to_df(folder, filenames):
    df = pd.DataFrame()
    for name in filenames:
        path = f'{folder}\\{name}'
        df = df.append(pd.read_table(path), ignore_index=True)
    return df
