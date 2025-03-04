# 事例集

- [事例集](#事例集)
  - [git push できない](#git-push-できない)
  - [ファイルの容量が大きすぎて怒られる](#ファイルの容量が大きすぎて怒られる)


## git push できない

<details><summary>エラーメッセージ</summary>

```bash
$ git push origin main
Missing or invalid credentials.
Error: connect ECONNREFUSED /run/user/1029/vscode-git-8ba35fb726.sock
    at PipeConnectWrap.afterConnect [as oncomplete] (node:net:1607:16) {
  errno: -111,
  code: 'ECONNREFUSED',
  syscall: 'connect',
  address: '/run/user/1029/vscode-git-8ba35fb726.sock'
}
Missing or invalid credentials.
Error: connect ECONNREFUSED /run/user/1029/vscode-git-8ba35fb726.sock
    at PipeConnectWrap.afterConnect [as oncomplete] (node:net:1607:16) {
  errno: -111,
  code: 'ECONNREFUSED',
  syscall: 'connect',
  address: '/run/user/1029/vscode-git-8ba35fb726.sock'
}
remote: Repository not found.
fatal: Authentication failed for 'https://github.com/sh-zoeee/zoe.git/'
```
</details>

なぜかHTTPS通信になっている。（このあたりの仕様はよくわからない）
基本的にSSHを使いたいので、以下のコマンドを打つ。
```bash
$ git remote set-url origin git@github.com:sh-zoeee/zoe.git
```
としてから`git push origin main`とすればよい


## ファイルの容量が大きすぎて怒られる

<details><summary>エラーメッセージ</summary>

```bash
$ git push -u origin main
Enumerating objects: 622, done.
Counting objects: 100% (622/622), done.
Delta compression using up to 64 threads
Compressing objects: 100% (440/440), done.
Writing objects: 100% (604/604), 38.92 MiB | 3.75 MiB/s, done.
Total 604 (delta 171), reused 587 (delta 158), pack-reused 0
remote: Resolving deltas: 100% (171/171), completed with 10 local objects.
remote: error: Trace: 53c961cd58abaee487267f6698a9ba4b4664026c5c270457128df17db83e3034
remote: error: See https://gh.io/lfs for more information.
remote: error: File data_cross/upos/English-EWT_English-EWT/tensor_2.pt is 196.86 MB; this exceeds GitHub's file size limit of 100.00 MB
remote: error: File data_cross/upos/French-GSD_Japanese-BCCWJ/tensor_3.pt is 639.54 MB; this exceeds GitHub's file size limit of 100.00 MB
remote: error: File data_cross/upos/Japanese-BCCWJ_Korean-Kaist/tensor_1.pt is 715.07 MB; this exceeds GitHub's file size limit of 100.00 MB
remote: error: File data_cross/upos/Japanese-BCCWJ_Korean-Kaist/tensor_3.pt is 715.07 MB; this exceeds GitHub's file size limit of 100.00 MB
remote: error: File data_cross/upos/Chinese-GSD_Japanese-BCCWJ/tensor_5.pt is 471.58 MB; this exceeds GitHub's file size limit of 100.00 MB
remote: error: File data_cross/upos/English-EWT_En-EWT-spacy/tensor_2.pt is 309.61 MB; this exceeds GitHub's file size limit of 100.00 MB
remote: error: GH001: Large files detected. You may want to try Git Large File Storage - https://git-lfs.github.com.
To github.com:sh-zoeee/zoe.git
 ! [remote rejected] main -> main (pre-receive hook declined)
error: failed to push some refs to 'github.com:sh-zoeee/zoe.git'
```
</details>

1つのファイルで100MBを超えるとpushのタイミングで怒られる。
そのファイルを履歴からも消す必要がある。
エラーとなっているファイルそれぞれについて、以下を実行する。
`data_cross/upos/French-GSD_Japanese-BCCWJ/tensor_3.pt`に対してなら、

```bash
$ git filter-branch --index-filter 'git rm --cached --ignore-unmatch data_cross/upos/French-GSD_Japanese-BCCWJ/tensor_3.pt' HEAD
$ git for-each-ref --format="%(refname)" refs/original/ | xargs -n 1 git update-ref -d
$ git reflog expire --expire=now --all
$ git gc --prune=now
```

を実行すればよい。1つ目のファイルのパスを変えてやればOK。
