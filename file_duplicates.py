import os

# load all file names from sub folders
# check for duplicates
root_folder = r"G:"

filenames = []
fullnames = []
for root, dirs, files in os.walk(root_folder):
    filenames.extend(files)
    fullnames.extend([(root, f) for f in files])

def gen_dupes(iterable):
    seen = []
    for idx, x in enumerate(iterable):
        if x in seen:
            yield idx, x
        seen.append(x)

for idx, file in gen_dupes(filenames):
    idx2 = filenames.index(filenames[idx])
    p1 = os.path.getsize(os.path.join(*fullnames[idx]))
    p2 = os.path.getsize(os.path.join(*fullnames[idx2]))
    if p1==p2:
        print(filenames[idx])

