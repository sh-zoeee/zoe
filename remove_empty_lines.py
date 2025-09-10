#!/usr/bin/env python3
import sys

def remove_empty_lines(input_file, output_file):
    """ファイルから空行を削除する"""
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 空行を削除（空白文字のみの行も含む）
    non_empty_lines = [line for line in lines if line.strip()]
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(non_empty_lines)
    
    print(f"元のファイル: {len(lines)} 行")
    print(f"空行削除後: {len(non_empty_lines)} 行")
    print(f"削除された空行数: {len(lines) - len(non_empty_lines)} 行")

if __name__ == "__main__":
    input_file = "/home/yamazoe/zoe/tmp/llm/GPT_ouput/gpt3.txt"
    output_file = "/home/yamazoe/zoe/tmp/llm/GPT_ouput/gpt3_no_empty_lines.txt"
    
    remove_empty_lines(input_file, output_file)
    print(f"処理完了: {output_file}")
