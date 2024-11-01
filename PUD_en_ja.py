from scripts import w_pq

def main():
    train_tensors_path = "data/train_tensors_ja_en_pud.pt"
    train_labels_path = "data/train_labels_ja_en_pud.pt"
    train_indexes_path = "data/train_indexes_ja_en_pud.pt"

    valid_tensors_path = "data/valid_tensors_ja_en_pud.pt"
    valid_labels_path = "data/valid_labels_ja_en_pud.pt"
    valid_indexes_path = "data/valid_indexes_ja_en_pud.pt"

    test_tensors_path = "data/test_tensors_ja_en_pud.pt"
    test_labels_path = "data/test_labels_ja_en_pud.pt"
    test_indexes_path = "data/test_indexes_ja_en_pud.pt"

    model_path = "models/weighted_distance_model_pud_en_ja.pth"


    TENSORS_PATH_LIST = [
        train_tensors_path, train_labels_path, train_indexes_path,
        valid_tensors_path, valid_labels_path, valid_indexes_path,
        test_tensors_path, test_labels_path, test_indexes_path
    ]

    CORPORA =   ["PUD/Japanese-PUDLUW.conllu", "PUD/English-PUD.conllu"]
    LABELS = ["ja", "en"]

    w_pq.preprocessing(
        CORPUS_FILE=CORPORA, type_of_labels=LABELS, save_file_list=TENSORS_PATH_LIST
    )

    print('\npreprocess finished.')

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