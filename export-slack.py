import json
import os
import time

from multiprocessing.dummy import Pool as ThreadPool

import urllib.request

rootdir = r"C:\Users\Luke A Sargen\Documents\UAS\Penn State UAS Slack export Jun 16 2015 - Sep 6 2022"
out_folder = r"C:\Users\Luke A Sargen\Documents\UAS\slack_export"

def parse_json(filename):
    with open(filename, encoding="utf-8") as f:
        data = json.load(f)

    for msg in data:
        if type(msg)==dict and "files" in msg.keys():
            for file in msg["files"]:
                if "url_private_download" in file.keys():
                    public_url = file["url_private_download"]
                    idx = public_url.find("?t=xoxe")
                    short_url = public_url[:idx]
                    basename = os.path.basename(short_url)
                    name, ext = os.path.splitext(basename)
                    # print(f"{public_url=}")
                    # print(f"{short_url=}")
                    # print(f"{basename=}")
                    out_filename = os.path.join(out_folder, f"{file['timestamp']}_{name}{ext}")
                    if not os.path.exists(out_filename):
                        urllib.request.urlretrieve(url=public_url, filename=out_filename)


# get all the jsons
jsons = []
for root, dirs, files in os.walk(rootdir):
    for f in files:
        jsons.append(os.path.join(root,f))

print(len(jsons))
start_idx = 772

# for idx, filename in enumerate(jsons[start_idx:]):
#     print(f"{start_idx+idx=}")
#     parse_json(filename)
    # break

t0 = time.time()

threads = 4
pool = ThreadPool(threads)
pool.map(parse_json, jsons)

dt = time.time()-t0
print(f"{dt=}")

