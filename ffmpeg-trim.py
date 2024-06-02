# used a list of in and out times
# saved to timestamps.txt
# created a safe file naem to input, quotations around quotations
# subprocces call

import os
import subprocess

# os.chdir('F:')
root = r"C:\Users\lukeasargen\Videos\youtube hhh"
os.chdir(root)
# subprocess.Popen("ls", cwd=".")

input_filename = "WORKING OUT AT VENICE BEACH [Z5oZva8DCRM].webm"
name, ext = os.path.splitext(input_filename)

# print(input_filename)
input_filename_safe = input_filename.replace("'","'\\''")
# print(input_filename_safe)
# exit()

trim_timestamp_list = [
# ["00:00:00.000", "00:00:04.599"],
# ["00:03:12.459", "00:03:42.982"],
# ["00:05:07.635", "00:05:38.163"],
# ["00:08:28.790", "00:08:34.985"],
# ["00:12:24.886", "00:13:14.122"],
# ["00:16:27.447", "00:16:59.380"],
# ["00:18:58.300", "00:19:07.968"],
# ["00:22:04.101", "00:22:13.455"],
# ["00:04:51.075", "00:05:12.795"],

]

for idx, (ss, to) in enumerate(trim_timestamp_list):

    output_filename = f"{name}-trim-{idx:03d}{ext}"
    output_filename_safe = output_filename.replace("'","'\\''")

    cmd_list = [
    'ffmpeg',
    '-threads', '4',
    '-ss', ss,
    '-to', to,
    '-i', input_filename,
    '-c', 'copy',
    output_filename,
    '-y'
    ]

    ret = subprocess.call(cmd_list)
# print(ret.returncode)
# print(ret)
