# 事例集

## git push できない

**エラーメッセージ**

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
なぜかHTTPS通信になっている。
基本的にSSHを使いたいので、以下のコマンドを打つ。
```bash
$ git remote set-url origin git@github.com:sh-zoeee/zoe.git
```
としてから`git push origin main`とすればよい