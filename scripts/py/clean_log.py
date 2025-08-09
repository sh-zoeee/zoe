from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--input", "-i", type=str, help="入力ログファイル")
parser.add_argument("--output", "-o", type=str, help="出力ファイル")
args = parser.parse_args()

input_path = args.input
output_path = args.output

with open(input_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

with open(output_path, "w", encoding="utf-8") as f:
    for line in lines:
        if line[0] == '#' or line[:4]=="STEP" or line[:3]=="ETA":
            continue  # この行はスキップ
        f.write(line)