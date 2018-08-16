---
layout: post
title:  "pytorch - gluon 함수 Mapping"
author: seujung
date:   2018-04-01 10:41:00
tags:	[mxnet,gluon]
image: /files/covers/dices.jpg
---

## 목적
---
- Deep learning 관련 Open Framework 중 mxnet(gluon) 으로 구현 시에 보통은 pytorch 코드를 참조하여서 구현을 많이 하고 있음(gluon user 부족으로 인해...)
- 대부분의 구조가 거의 비슷하지만 일부 함수의 경우에는 내용이 달라서 pytorch 함수를 gluon 함수로 변환하는 과정이 필요함


## 함수 mapping 표 (update 진행 중)
---


|  <center>Function</center> | <center>Pytorch</center> |  <center>Gluon</center> |<center>비고</center> |
|:--------|:--------|:--------|:--------|
|Dimension 삽입|torch.unsqueeze(data,1)| data.expand_dims(1)|
|Reshape|x.view(1,2,3)| x.reshape(shape=(1,2,3))||
|Swap shape|x.permute(0,2,1)|x.swapaxes(1,2)|F.swapaxes(x,1,2)는 deprecated 됨|
|Data Copy|x.repeat(1,25,1,1)|F.repeat(x,repeats=25,axis=1)||
|Concat data|torch.cat([a,b],3)|F.concat(a,b,dim=3)||
|Return specific shape |x.size()[2]|x.shape[2]||
|Batch matrix product |x.bmm(y)|nd.linalg_gemm2(x, y)||
|Clipping |x.clamp(min, max)|nd.clip(x, min, max)||
|Convert to numpy|x.numpy()|x.asnumpy()||


## Reference
[PyTorch to MXNET](https://github.com/zackchase/mxnet-the-straight-dope/blob/master/cheatsheets/pytorch_gluon.md)