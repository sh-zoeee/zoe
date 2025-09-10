# show_err_head.py
from openai import OpenAI
client = OpenAI()
ERR_ID = "file-FjAdc1SS24UZbbM5fjVX2s"  # ←ログに出ていたID
txt = client.files.content(ERR_ID).content.decode("utf-8", errors="replace")
print("\n".join(txt.splitlines()[:80]))
