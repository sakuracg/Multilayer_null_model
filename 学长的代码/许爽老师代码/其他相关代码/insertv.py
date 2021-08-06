# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 21:28:43 2020

@author: Administrator
"""

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

'''
threesituations
'''
#image = Image.open("E:\\111111111111111111111111111111\\new\\论文\\零模型重置网络\\3c\\threesituations.png")
#
#draw = ImageDraw.Draw(image)
## 设置字体和大小
#font = ImageFont.truetype('msyh.ttf', 100)
## 定义字体颜色
#fillcolor = "#000000"
## 获取图片的长宽属性
#width, height = image.size
## 设置字体添加到图片中的位置,即对图片进行操作
#draw.text((width*0.18, height*0.18), '0.044', font=font, fill=fillcolor)
#draw.text((width*0.18, height*0.28), '0.015', font=font, fill=fillcolor)
## 将修改后的图片保存，保存格式设置为jpeg
#image.save('E:\\111111111111111111111111111111\\new\\论文\\零模型重置网络\\3c\\threesituations标注.png', 'png')

'''
foursituations
'''
#
image = Image.open("E:\\111111111111111111111111111111\\new\\论文\\零模型重置网络\\3c\\时变网络的接触置乱算法.png")

draw = ImageDraw.Draw(image)
# 设置字体和大小
font = ImageFont.truetype('msyh.ttf', 100)
# 定义字体颜色
fillcolor = "#000000"
# 获取图片的长宽属性
width, height = image.size
# 设置字体添加到图片中的位置,即对图片进行操作
draw.text((width*0.18, height*0.115), '0.028', font=font, fill=fillcolor)
draw.text((width*0.18, height*0.20), '0.023', font=font, fill=fillcolor)
draw.text((width*0.18, height*0.28), '0.008', font=font, fill=fillcolor)
# 将修改后的图片保存，保存格式设置为jpeg
image.save('E:\\111111111111111111111111111111\\new\\论文\\零模型重置网络\\3c\\时变网络的接触置乱算法标注.png', 'png')