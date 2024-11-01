from scripts import w_pq
from tqdm import tqdm
import torch
from sklearn.model_selection import train_test_split


def test(
        train_tensors_file: str, train_labels_file: str,
        test_tensors_file: str, test_labels_file: str,
        model_path: str
        ):
    
    train_tensors = torch.load(train_tensors_file)
    train_labels = torch.load(train_labels_file)

    test_tensors = torch.load(test_tensors_file)
    test_labels = torch.load(test_labels_file)

    dimension_tensor = test_tensors[0].size()[0]

    # モデルと損失関数 
    distance_function = w_pq.WeightedPqgramDistance(dimension_tensor)
    distance_function.load_state_dict(torch.load(model_path))
    distance_function.eval().to("cuda:0")

    #train_tensors, _, train_labels, _ = train_test_split(
    # train_tensors, train_labels, random_state=42, shuffle=True, train_size=2000
    #)

    
    num_test = len(test_labels)
    error = 0

    k = 1
    
    
    for i in tqdm(range(100), desc="[test  loop]"):
        
        test_tensor = test_tensors[i].to("cuda:0")
        
        predicted_label = w_pq.predict_label_1nn(
            distance_function, test_tensor, train_tensors, train_labels
        )

        correct_label = test_labels[i]

        correct_ewt1 = 0
        incorrect_ewt1 = 0
        correct_ewt2 = 0
        incorrect_ewt2 = 0

        if correct_label == "EWT1":
            if predicted_label == "EWT1":
                correct_ewt1 += 1
            elif predicted_label == "EWT2":
                incorrect_ewt1 += 1
            else:
                print("predicted label is wrong!")
                print(predicted_label)
                exit()

        elif correct_label == "EWT2":
            if predicted_label == "EWT1":
                incorrect_ewt2 += 1
            elif predicted_label == "EWT2":
                correct_ewt2 += 1
            else:
                print("predicted label is wrong!")
                print(predicted_label)
                exit()
        else :
            print("correct label is wrong!")
            print(correct_label)
            exit()

    
    print(f'\terror rate: {(correct_ewt1+correct_ewt2)/100:.2f}')




def main():
    train_tensors_path = "data/train_tensors_en_corpora_EWT_EWT_upos.pt"
    train_labels_path = "data/train_labels_en_corpora_EWT_EWT_upos.pt"
    train_indexes_path = "data/train_indexes_en_corpora_EWT_EWT_upos.pt"

    valid_tensors_path = "data/valid_tensors_en_corpora_EWT_EWT_upos.pt"
    valid_labels_path = "data/valid_labels_en_corpora_EWT_EWT_upos.pt"
    valid_indexes_path = "data/valid_indexes_en_corpora_EWT_EWT_upos.pt"

    test_tensors_path = "data/test_tensors_en_corpora_EWT_EWT_upos.pt"
    test_labels_path = "data/test_labels_en_corpora_EWT_EWT_upos.pt"
    test_indexes_path = "data/test_indexes_en_corpora_EWT_EWT_upos.pt"

    model_path = "models/model_en_corpora_EWT_EWT_upos.pth"

    TENSORS_PATH_LIST = [
        train_tensors_path, train_labels_path, train_indexes_path,
        valid_tensors_path, valid_labels_path, valid_indexes_path,
        test_tensors_path, test_labels_path, test_indexes_path
    ]

    CORPORA =   ["corpora/English-EWT.conllu", "corpora/English-EWT.conllu"]
    LABELS = ["EWT1", "EWT2"]

    
    
    test(
        train_tensors_path, train_labels_path,
        test_tensors_path, test_labels_path, 
        model_path
        )

    print('\ntest finished.')
    

if __name__=="__main__":
    main()

