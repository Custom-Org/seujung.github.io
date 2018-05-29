---
layout: post
title:  "[Papar Review] Relational Network Review"
author: seujung
date:   2018-05-29 10:11:31
tags:	[deep-learning,paper_review]
image: /files/covers/deeplearning-cover.jpg
---

## 논문 개요
---
 Visual QA 문제에서 관계형 추론 (relational reasoning) 이란 방법을 제시하여서 높은 성능을 보인 Deep Mind 논문에 대해서 설명하고자 합니다. 상당히 간단한 구조의 알고리즘을 통해 기존 방법과 확연한 차이를 보인 점이 가장 큰 장점입니다. 또한 해당 논문에서 제시한 방법은 QA 문제를 넘어서 다양한 Domain에서도 활용할 수 있다는 점이 또 하나의 큰 장점 입니다.

<https://arxiv.org/pdf/1706.01427.pdf>

## 문제 사항 설명
해당 논문에서 주요 다룬 데이터는 [CLEVR](https://cs.stanford.edu/people/jcjohns/clevr/) data set을 사용 하였습니다. 해당 데이터의 주요 구조는 다음과 같습니다.

![fig1](/files/180529_rl_model/fig1.png)

Original image 데이터를 보고 text로 구성된 질문에 대한 답을 제시하는 형태입니다. 질문은 크게 2가지 형태로 구성되오 있습니다.
  - Relational question : 이미지 안에 있는 object 간의 관계를 파악하여서 정답을 제시하는 구조
  - Non-relational question : 이미지 안에 있는 하나의 object의 특성을 제시하는 구조


## Network Architecture
---

relational network은 다음의 수식 형태로 표현할 수 있습니다.


$$ RN(O) = f_\phi\left(\sum_{i,j} g_\theta(o_i, o_j)\right) $$

데이터 형태를 object로 변환한 다음에 각각의 object를 input으로 하는 함수 $ g_\theta $ 에 대입하고 난 후 그 결과를 합한 이후에 $ f_\phi $ 에 다시 통과하는 형태 입니다.

해당 수식을 그림으로 표헌하면 다음과 같습니다.
