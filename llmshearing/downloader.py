import gdown
import os

with open('urls', 'r') as f:
    data = f.read()

urls = [url.strip() for url in data.split(',')][49:]

cnt = 49

def gen_name(cnt):
    if cnt < 100:
        return os.path.join("/remote-home/rypeng/0224/LLM-Shearing/data/for_prune/cc", f"shard.000{cnt}.mds")
    else:
        return os.path.join("/remote-home/rypeng/0224/LLM-Shearing/data/for_prune/cc", f"shard.00{cnt}.mds")


for url in urls:
    output = gen_name(cnt)
    cnt += 1
    if os.path.exists(output):
        continue
    while True:
        try:
            gdown.download(
                url, output, proxy="http://10.176.52.116:7890", fuzzy=True
            )
            break
        except:
            continue
