from scripts import w_pq_batch as w_pq

import time, torch



def main():

    random_state = 50
    p = 2
    q = 2

    torch.cuda.empty_cache()
    

    train_tensors_path = f"data/train_tensors_en_corpora_En-EWT_Ja-BCCWJ_upos_{str(random_state)}.pt"
    train_labels_path = f"data/train_labels_en_corpora_En-EWT_Ja-BCCWJ_upos_{str(random_state)}.pt"
    train_indexes_path = f"data/train_indexes_en_corpora_En-EWT_Ja-BCCWJ_upos_{str(random_state)}.pt"

    valid_tensors_path = f"data/valid_tensors_en_corpora_En-EWT_Ja-BCCWJ_upos_{str(random_state)}.pt"
    valid_labels_path = f"data/valid_labels_en_corpora_En-EWT_Ja-BCCWJ_upos_{str(random_state)}.pt"
    valid_indexes_path = f"data/valid_indexes_en_corpora_En-EWT_Ja-BCCWJ_upos_{str(random_state)}.pt"

    test_tensors_path = f"data/test_tensors_en_corpora_En-EWT_Ja-BCCWJ_upos_{str(random_state)}.pt"
    test_labels_path = f"data/test_labels_en_corpora_En-EWT_Ja-BCCWJ_upos_{str(random_state)}.pt"
    test_indexes_path = f"data/test_indexes_en_corpora_En-EWT_Ja-BCCWJ_upos_{str(random_state)}.pt"

    model_path = f"models/model_en_corpora_EWT_Ja-BCCWJ_{str(random_state)}.pth"

    TENSORS_PATH_LIST = [
        train_tensors_path, train_labels_path, train_indexes_path,
        valid_tensors_path, valid_labels_path, valid_indexes_path,
        test_tensors_path, test_labels_path, test_indexes_path
    ]

    CORPORA =   ["corpora/English/English-EWT.conllu", "corpora/Japanese/Japanese-BCCWJ.conllu"]
    LABELS = ["EWT","Ja-BCCWJ"]

    start = time.time()

    w_pq.preprocessing(
        CORPUS_FILE=CORPORA, 
        type_of_labels=LABELS, 
        save_file_list=TENSORS_PATH_LIST, 
        random_state=random_state,
        p = p,  q = q,
    )
    
    
    
    print('\npreprocess finished.')

    w_pq.train(
        train_tensors_path, train_labels_path, 
        valid_tensors_path, valid_labels_path,
        model_path, loss_figure_path=f"figures/En-EWT_Ja-BCCWJ_upos_{str(random_state)}.png"
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

