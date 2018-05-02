# -*- coding: utf-8 -*-
"""
Created on Tue May  1 18:09:29 2018

@author: liying30
"""

#题目1:有1、2、3、4个数字，能组成多少个互不相同且无重复的两位数，都是多少
data = []
dataSource = [1,2,3,4]
for i in dataSource:
    i = i-1
    for j in dataSource:
        j = j-1
        if(dataSource[i] != dataSource[j]):
            dataIerm = 10*dataSource[i] +dataSource[j] 
            if dataIerm in data:
                pass
            else:
                data.append(dataIerm)
#输入三个整数x，y，z，请把这三个数由小到达输出，可掉用input(),(需要加判断，判断输入数据是否为数字)
x = input()
y = input()
z = input()
sortList(x,y,z)
def sortList(x,y,z):
    if not(x.isdigit() & y.isdigit() & z.isdigit()):
        return
    if  int(x) > int(y) :
        tmp = x
        x = y
        y = tmp
    if int(y) > int(z):
        tmp = y
        y = z
        z = tmp
    if int(x) > int(y):
        tmp = x
        x = y
        y = tmp
    print(x,y,z)
        
        
#题目3：输入一行字符，分别统计其中英文字母、空格、数字和其它字符的个数
data = input()
alpNum = 0
digitNum = 0
otherNum = 0
for l in data:
    if(l.isalpha()):
        alpNum = alpNum+1
    if(l.isdigit()): 
        digitNum = digitNum +1
    else:
        otherNum = otherNum +1

#
#猴子第一天摘下若干个桃子，当即吃了一半，还不过瘾，又吃了一个
#第二天早上又将剩下的桃子吃掉一半，又多吃了一个
#以后每天早上都吃前一天声息的一半零一个。到第十天早上想再吃，只剩下一个桃子，求第一天共摘了多少个

#1
#x+1
#2*(x+2) = 2*x+4
#2*(2*x+5) = 4*x+10
#2*(4*x+11) = 8*x +22
#2*(8*x +23) = 16*x +46
#2*(16*x +47) = 32*x + 94
#2*(32*x + 95) = 64*x +190
#2*(64*x +191) = 128*x +382
#2*( 128*x +383) = 256*x + 766


#x = 3
#n = 1534

x = 3
i = 0
while(i<8):
    x = 2*(x+1)
    x = x+1
    i = i+1
n = x+1    

#猜数问题，要求如下
#1、随机生成一个整数
#2、猜一个数字并输入
#判断时间，需要用到time和random模块

import random
import time
num = random.randint(0,999)
a =int(input())
start = int(time.time())
while(a != num):
    if(a < num):
        print('num is small')
    else:
        print('num is large') 
    a =int(input())
useTime = int(time.time()) - start
print('you have use '+str(useTime)+' time')