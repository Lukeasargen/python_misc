import os
import subprocess

# os.chdir('F:')
root = r"C:\Users\lukeasargen\Documents\source"
os.chdir(root)
# subprocess.Popen("ls", cwd=".")

input_filename = "input.mp4"
name, ext = os.path.splitext(input_filename)
output_filename = f"{name}-trimmed{ext}"

# https://calculateaspectratio.com/
# 1920 1080 604
height = 1080
width = 1080
x = (1920/2 - width/2)
y = 0

# 1280 720 406
# height = 720
# width = 400
# x = (1280/2 - width/2)
# y = 0

# 480 270
# height = 480
# width = 270
# x = (854/2 - width/2)
# y = 0

# height = 720
# width = 600
# x = 420
# y = 0

safe_filename = input_filename.replace("'","'\\''")
# print(safe_filename)

cmd_list = [
'ffmpeg',
'-threads',
'4',
'-ss',
'00:00:03.200',
# '-to',
# '00:00:05.000',
'-i',
input_filename,
'-filter:v',
f'crop={width}:{height}:{x}:{y}',
# f'crop={width}:{height}:{x}:{y},transpose=3',
# f'crop={width}:{height}:{x}:{y},scale=960:720,setsar=1',
# f'scale=720:480,setsar=1',
# "-vf",
# "transpose=3",

output_filename,
'-y'
]

ret = subprocess.call(cmd_list)
# print(ret.returncode)
# print(ret)