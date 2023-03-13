from octis.preprocessing.preprocessing import Preprocessing
import string
def reencode_and_write_to_single_file(texts, file_name):
    txt_file = f'.\{file_name}.txt'
    text = "\n".join(texts)
    text = text.encode('cp1252', 'ignore').decode('cp1252', 'ignore')
    with open(txt_file, mode='wt') as file:
        file.write(text)


def get_basic_preprocessor():
    preprocessor = Preprocessing(
                lowercase=False,
                remove_punctuation=False,
                punctuation=string.punctuation,
                remove_numbers=False,
                lemmatize=False,
                language="english",
                split=False,
                verbose=True,
                save_original_indexes=True,
                remove_stopwords_spacy=False,
            )
    return preprocessor

def get_lda_preprocessor():
    preprocessor = Preprocessing(
            lowercase=True,
            remove_punctuation=True,
            punctuation=string.punctuation,
            remove_numbers=True,
            lemmatize=True,
            language="english",
            split=False,
            verbose=True,
            save_original_indexes=True,
            stopword_list='english',
        )
    return preprocessor
    
def create_octis_dataset(text_file, preprocessor, output_folder):
    dataset = preprocessor.preprocess_dataset(documents_path=text_file)
    dataset.save(output_folder)

def reencode_vocab(octis_folder):
    vocab = octis_folder+'/vocabulary.txt'
    with open(vocab, encoding='utf-8') as f:
        content = f.read()
    content = content.encode('cp1252', 'ignore').decode('cp1252', 'ignore')
    with open(vocab, mode='wt') as file:
        file.write(content)

def create_lda_and_multimodal_dataset(text_file, output_folders={'lda':'lda_custom__octis_dataset',
                                                                  'multimodal':'multimodal_custom_octis_dataset'}
                                    ):
    lda_preprocessor = get_lda_preprocessor()
    multimodal_preprocessor = get_basic_preprocessor()
    create_octis_dataset(text_file, lda_preprocessor, output_folders['lda'])
    create_octis_dataset(text_file, multimodal_preprocessor, output_folders['multimodal'])
    reencode_vocab(output_folders['lda'])
    reencode_vocab(output_folders['multimodal'])