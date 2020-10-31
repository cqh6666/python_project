import requests
from bs4 import BeautifulSoup
import os
import shutil


urls = "https://music.163.com/artist?id=3684"
headers = {
    "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.111 Safari/537.36 Edg/86.0.622.56"
}
PATH = "../英语pre"
NEW_PATH = "../英语pre/歌词更新"
def request_songs(urls):
    s = requests.session()
    bs = BeautifulSoup(s.get(urls, headers=headers).content, "lxml")

    artlist = bs.find('ul',class_="f-hide")
    songs = artlist.find_all('a')

    # f = open('song_id.txt', mode='w')
    for song in songs:
        href = song['href']
        name = song.get_text()
        href2 = href.split("=")[1]
        print(href2,name)
        # f.write(href2+",")

    # f.close()


def process_song(file):
    file_name = os.path.splitext(file)[0]
    if os.path.splitext(file)[1] != '.txt':
        return

    with open(file,'r',encoding='utf-8') as f1, open(file_name+"_update.txt",'w',encoding='utf-8') as f2:
        for line in f1:
            line = line.split("]")[1]
            str = "："
            str2 = ":"
            str3 = "ISRC"
            if str not in line and str2 not in line and str3 not in line and line:
                f2.write(line)

        print("成功修改", file_name)

def process_all_song(pathname):
    files = os.listdir(pathname)
    for filename in files:
        process_song(os.path.join(pathname, filename))

def move_file(oldpath,newpath):
    files = os.listdir(oldpath)
    for filename in files:
        str = "update"
        if str in os.path.splitext(filename)[0]:
            shutil.move(os.path.join(PATH,filename), newpath)

#
# process_all_song(PATH)
# move_file(PATH,NEW_PATH)