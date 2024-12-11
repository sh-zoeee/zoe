from scripts import w_pq_batch as w_pq

import time, torch



def main():

    import argparse
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--source", "-s", help="比較元")
    argparser.add_argument("--target", "-t", help="言語")
    args = argparser.parse_args()


    random_state = 50
    label_type = "unlabel"
    p = 2
    q = 2
    source = args.source
    target = args.target

    torch.cuda.empty_cache()
    

    train_tensors_path = f"data/train_tensors_{source}_{target}_{label_type}_{str(random_state)}.pt"
    train_labels_path = f"data/train_labels_{source}_{target}_{label_type}_{str(random_state)}.pt"
    train_indexes_path = f"data/train_indexes_{source}_{target}_{label_type}_{str(random_state)}.pt"

    valid_tensors_path = f"data/valid_tensors_{source}_{target}_{label_type}_{str(random_state)}.pt"
    valid_labels_path = f"data/valid_labels_{source}_{target}_{label_type}_{str(random_state)}.pt"
    valid_indexes_path = f"data/valid_indexes_{source}_{target}_{label_type}_{str(random_state)}.pt"

    test_tensors_path = f"data/test_tensors_{source}_{target}_{label_type}_{str(random_state)}.pt"
    test_labels_path = f"data/test_labels_{source}_{target}_{label_type}_{str(random_state)}.pt"
    test_indexes_path = f"data/test_indexes_{source}_{target}_{label_type}_{str(random_state)}.pt"

    model_path = f"models/model_{source}_{target}_{label_type}_{str(random_state)}.pth"

    TENSORS_PATH_LIST = [
        train_tensors_path, train_labels_path, train_indexes_path,
        valid_tensors_path, valid_labels_path, valid_indexes_path,
        test_tensors_path, test_labels_path, test_indexes_path
    ]

    CORPORA =   [f"corpora/Korean/Korean-Kaist.conllu", f"corpora/Japanese/Japanese-BCCWJ.conllu"]
    #CORPORA =   [f"PUD/{source}-PUD.conllu", f"PUD/{target}-PUD.conllu"]
    LABELS = [0,1]

    start = time.time()

    w_pq.preprocessing(
        CORPUS_FILE=CORPORA, 
        type_of_labels=LABELS, 
        save_file_list=TENSORS_PATH_LIST, 
        random_state=random_state,
        p = p,  q = q,
        label_type = label_type
    )
    
    
    
    print('\npreprocess finished.')

    w_pq.train(
        train_tensors_path, train_labels_path, 
        valid_tensors_path, valid_labels_path,
        model_path, loss_figure_path=f"figures/parser_{source}_{target}_{str(random_state)}.png"
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

