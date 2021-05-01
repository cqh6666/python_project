import matplotlib.pyplot as plt
import jieba
import os
from wordcloud import WordCloud
import numpy as np
from PIL import Image

# https://blog.csdn.net/fly910905/article/details/77763086/

songs_path = '../歌词更新/'
other_path = '../result'

def get_all_songs(path):
    files = os.listdir(path)

    for file in files:
        with open("../result/songs2.txt", 'a', encoding='utf-8') as fn, open(os.path.join(path, file), 'r', encoding='utf-8') as f2:
            for line in f2:
                fn.write(line)
        print(file,"插入成功")



# get_all_songs(songs_path)

text = open(os.path.join(other_path, "songs2.txt"),encoding='utf-8').read()
cut_text = jieba.cut(text)
result = ",".join(cut_text)

color_mask = np.array(Image.open(os.path.join(other_path, "图片4.png")))

wc = WordCloud(
    # 设置字体，不指定就会出现乱码
    font_path=os.path.join(other_path, "msyh.ttc"),
    # 是否透明
    mode="RGBA",
    # 背景图
    mask=color_mask,
    # 设置背景色
    background_color='white',
    # # 设置背景宽
    # width=500,
    # # 设置背景高
    # height=350,
    max_words =150,
    # 最大字体
    max_font_size=120,
    # 最小字体
    min_font_size=20,
)  # max_words=1000 ),mode='RGBA',colormap='pink')
# 产生词云
wc.generate(result)
# 保存图片
wc.to_file(os.path.join(other_path, "cloudword4.png"))  # 按照设置的像素宽高度保存绘制好的词云图，比下面程序显示更清晰
# 4.显示图片
# 指定所绘图名称
plt.figure("song")
# 以图片的形式显示词云
plt.imshow(wc)
# 关闭图像坐标系
plt.axis("off")
plt.show()
