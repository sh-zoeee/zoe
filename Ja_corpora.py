from scripts import w_pq


def main():
    train_tensors_path = "data/train_tensors_ja_corpora.pt"
    train_labels_path = "data/train_labels_ja_corpora.pt"
    train_indexes_path = "data/train_indexes_ja_corpora.pt"

    valid_tensors_path = "data/valid_tensors_ja_corpora.pt"
    valid_labels_path = "data/valid_labels_ja_corpora.pt"
    valid_indexes_path = "data/valid_indexes_ja_corpora.pt"

    test_tensors_path = "data/test_tensors_ja_corpora.pt"
    test_labels_path = "data/test_labels_ja_corpora.pt"
    test_indexes_path = "data/test_indexes_ja_corpora.pt"

    model_path = "models/model_ja_corpora.pth"

    TENSORS_PATH_LIST = [
        train_tensors_path, train_labels_path, train_indexes_path,
        valid_tensors_path, valid_labels_path, valid_indexes_path,
        test_tensors_path, test_labels_path, test_indexes_path
    ]

    CORPORA =   ["corpora/Japanese-BCCWJ.conllu", "corpora/Japanese-GSDLUW.conllu", "corpora/Japanese-Modern.conllu"]
    LABELS = ["BCCWJ", "GSDLUW", "Modern"]

    """
    w_pq.preprocessing(
        CORPUS_FILE=CORPORA, type_of_labels=LABELS, save_file_list=TENSORS_PATH_LIST
    )

    print('\npreprocess finished.')"""
   
    w_pq.train(
        train_tensors_path, train_labels_path, 
        valid_tensors_path, valid_labels_path,
        model_path
    )

    print('\ntrain finished.')
    
    
    w_pq.test(
        train_tensors_path, train_labels_path,
        test_tensors_path, test_labels_path, 
        model_path
        )

    print('\ntest finished.')

if __name__=="__main__":
    main()

