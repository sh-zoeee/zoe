from scripts import w_pq_batch as w_pq

import time, torch



def main():

    random_state = 20

    torch.cuda.empty_cache()
    start = time.time()

    train_tensors_path = "data/train_tensors_en_corpora_En_EWT_udpipe_unlabel_20.pt"
    train_labels_path = "data/train_labels_en_corpora_En_EWT_udpipe_unlabel_20.pt"
    train_indexes_path = "data/train_indexes_en_corpora_En_EWT_udpipe_unlabel_20.pt"

    valid_tensors_path = "data/valid_tensors_en_corpora_En_EWT_udpipe_unlabel_20.pt"
    valid_labels_path = "data/valid_labels_en_corpora_En_EWT_udpipe_unlabel_20.pt"
    valid_indexes_path = "data/valid_indexes_en_corpora_En_EWT_udpipe_unlabel_20.pt"

    test_tensors_path = "data/test_tensors_en_corpora_En_EWT_udpipe_unlabel_20.pt"
    test_labels_path = "data/test_labels_en_corpora_En_EWT_udpipe_unlabel_20.pt"
    test_indexes_path = "data/test_indexes_en_corpora_En_EWT_udpipe_unlabel_20.pt"

    model_path = "models/model_en_corpora_EWT_udpipe_unlabel_20.pth"

    TENSORS_PATH_LIST = [
        train_tensors_path, train_labels_path, train_indexes_path,
        valid_tensors_path, valid_labels_path, valid_indexes_path,
        test_tensors_path, test_labels_path, test_indexes_path
    ]

    CORPORA =   ["corpora/English-EWT.conllu", "parsed/En-EWT-udpipe.conllu"]
    LABELS = ["EWT", "udpipe"]


    w_pq.preprocessing(
        CORPUS_FILE=CORPORA, 
        type_of_labels=LABELS, 
        save_file_list=TENSORS_PATH_LIST, 
        random_state=random_state
    )
    
    
    
    print('\npreprocess finished.')

    w_pq.train(
        train_tensors_path, train_labels_path, 
        valid_tensors_path, valid_labels_path,
        model_path, loss_figure_path="figures/En_EWT_udpipe_unlabel_20.png"
    )

    print('\ntrain finished.')
    
    
    w_pq.test(
        train_tensors_path, train_labels_path,
        test_tensors_path, test_labels_path, 
        model_path
        )

    print('\ntest finished.')
    end = time.time()

    print(f'\ntime: {end-start} sec')
    
    

if __name__=="__main__":
    main()

