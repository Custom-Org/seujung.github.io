---
layout: post
title:  "pytorch - gluon 함수 Mapping"
author: seujung
date:   2018-04-01 10:41:00
tags:	[mxnet,gluon]
image: /files/covers/kage-idc.jpg
---

## 목적
---
- Deep learning 관련 Open Framework 중 mxnet(gluon) 으로 구현 시에 보통은 pytorch 코드를 참조하여서 구현을 많이 하고 있음(gluon user 부족으로 인해...)
- 대부분의 구조가 거의 비슷하지만 일부 함수의 경우에는 내용이 달라서 pytorch 함수를 gluon 함수로 변환하는 과정이 필요함


## 함수 mapping 표 (update 진행 중)
---
|  <center>Pytorch</center> |  <center>Gluon</center> |
|:--------|:--------:|
|torch.unsqueeze(data,1)| data.expand_dims(1)|
|x.view(1,2,3)| x.reshape(shape=(1,2,3))|
|x.permute(0,2,1)|F.swapaxes(x,1,2)|
|x.repeat(1,25,1,1)|F.repeat(x,repeats=25,axis=1)|
|torch.cat([a,b],3)|F.concat(a,b,dim=3)|
|x.size()[2]|x.shape[2]|
