# -*- encoding: utf-8 -*-
'''
@create_time: 2021/02/23 10:25:45
@author: lichunyu
'''

import PIL
from PIL import Image, ImageDraw, ImageFont
import math


def image_gen(width:int=0, height:int=0, words:list=None, coors:list=None, img_name:str='', \
                color_text:tuple=(0,0,0), color_bg:tuple=(255,255,255), channel:str='RGB', scale:float=0.9):
    img = Image.new(channel, (width, height), color_bg)
    draw = ImageDraw.Draw(img)
    for word, coor in zip(words, coors):
        text_size = math.ceil((coor[3] - coor[1]) * scale)
        fontStyle = ImageFont.truetype('Hiragino Sans GB.ttc', text_size, encoding="utf-8")
        draw.text((coor[0], coor[1]), word, color_text, font=fontStyle)
    img.save(img_name)
    img.close()
    return img_name


if __name__ == '__main__':
    import json
    with open('./tools/text.json', 'r') as f:
        jsondata = f.read()
    data = json.loads(jsondata)
    data_page = data['document_list_json'][0]['page_list'][0]
    height = data_page['height']
    width = data_page['width']
    words, coors = [], []
    for chunk in data_page['chunk_list']:
        words.append(chunk['text'])
        coors.append(chunk['coor'])
    img_name = image_gen(width=width, height=height, words=words, coors=coors, img_name='test.jpg')
    print('==========success===============')