---
layout: post
title:  "Relational Network Review"
author: seujung
date:   2018-05-29 10:11:31
tags:	[deep-learning,paper]
image: /files/covers/deeplearning-cover.jpg
---

#### 논문 개요
 Visual QA 문제에서 관계형 추론 (relational reasoning) 이란 방법을 제시하여서 높은 성능을 보인 Deep Mind 논문에 대해서 설명하고자 합니다. 상당히 간단한 구조의 알고리즘을 통해 기존 방법과 확연한 차이를 보인 점이 가장 큰 장점입니다. 또한 해당 논문에서 제시한 방법은 QA 문제를 넘어서 다양한 Domain에서도 활용할 수 있다는 점이 또 하나의 큰 장점 입니다.

<https://arxiv.org/pdf/1706.01427.pdf>

#### 문제 사항 설명
해당 논문에서 주요 다룬 데이터는 [CLEVR](https://cs.stanford.edu/people/jcjohns/clevr/) data set을 사용 하였습니다. 해당 데이터의 주요 구조는 다음과 같습니다.

![fig1](/files/180529_rl_model/fig1.png)

Original image 데이터를 보고 text로 구성된 질문에 대한 답을 제시하는 형태입니다. 질문은 크게 2가지 형태로 구성되오 있습니다.
  - Relational question : 이미지 안에 있는 object 간의 관계를 파악하여서 정답을 제시하는 구조
    - 위의 그림을 예로 들면 각각 object 간의 형태 및 특징을 파악 한 다음에 서로의 특징을 비교하는 형태의 질문을 의미합니다.(노란색 원통형 물체와 같은 물체가 있는가?)

  - Non-relational question : 이미지 안에 있는 하나의 object의 특성을 제시하는 구조
      - 위의 그림을 예로 들면 어떤 특정 물체의 모양/색깔/크기 등을 직접적으로 물어보는 질문을 의미합니다(갈색 공의 크기는 어떠한가?)


#### Network Architecture
---

relational network은 다음의 수식 형태로 표현할 수 있습니다.


$$ RN(O) = f_\phi\left(\sum_{i,j} g_\theta(o_i, o_j)\right) $$

데이터 형태를 object로 변환한 다음에 각각의 object를 input으로 하는 함수 $ g_\theta $ 에 대입하고 난 후 그 결과를 합한 이후에 $ f_\phi $ 에 다시 통과하는 형태 입니다.

해당 수식을 그림으로 표헌하면 다음과 같습니다.

![fig2](/files/180529_rl_model/fig2.png)

relational network에서 가장 중요한 것은 우리가 파악하고자 하는 대상을 domain에 상관없이 object라는 객체로 변환하여 해당 object 간의 관계를 파악하는 것입니다. 여기에서 이미지 데이터는 CNN을 통해 산출한 Feature maps을 질문의 경우에는 LSTM을 통과한 결과 값을 object로 간주합니다. 우리는 이미지와 문장 간의 관계가 아닌 object와 object 간의 관계를 파악하는 것으로 문제를 변환하게 됩니다.
관계를 파악 하는 방법은 의외로 간단합니다. 각 object 별로 concat을 수행한 후에  MLP($ g_\theta $) 의 input을 하게 됩니다. $ g_\theta $를 통과하고 나온 값에 대해 element-wise sum을 수행한 후 다시 한 번 MLP($ f_\phi $)를 수행하여서 최종적으로 우리가 알고 싶은 class에 대한 값을 산출하게 됩니다.

이 때 가장 중요한 점은  RN 부분에서의 input data size가 달라지는 점입니다. pair 개수만큼 새로 데이터를 생성하는 형태이기 때문에 $ pair^2 $ 만큼의 데이터가 증가하게 됩니다. 증가된 데이터가  $ g_\theta $ 를 통과하게 되고 증가한 만큼의 데이터를 원래의 size로 줄이기 위해 element-wise sum 을 수행하개 됩니다.


#### 수행 결과
성능은 기존의 알고리즘에 비해 월등히 좋은 것을 확인 할 수가 있습니다.
![fig3](/files/180529_rl_model/fig3.png)


#### Test 데이터 구조
CLEVR 데이터로 실험을 바로 수행하면 좋으나 Data Set 크기 및 학습 속도를 고러하여 CLEVR 데이터를 사용하지 않고 논문에서 제시한 Sort-of-CLEVR 데이터 셋을 사용 하였습다

#### Sort-of-CLVR 데이터 셋 구조
이미지 형태는 1 X 75 X 75 형태이며 각각 이미지당 20개의 질문(10 relational / 10 non-relational)으로 구성되어 있습니다.
질문의 형태는 다음과 같습니다.
- Non-relational Question Case
  - query shape : 빨강색 물체의 형태는?
  - query horiziontal position : 빨간색 물체 왼쪽에 물체가 있는가?
  - query vertical position : 빨간색 물체 위에 물체가 있는가?

- Relational Question Case
  - closest-to : 녹색 물체와 가장 가까운 물체의 형태는?
  - furthest-from : 녹색 물체와 가장 먼 물체의 형태는?
  - count : 녹색 물체는 총 몇 개인가?

질문에 대한 대답은 총 10가지 형태 입니다
  - color : red, green, black, orange, yellow, gray
  - Yes/No : yes , no
  - object 형테 : rectangle, circle


![Sort-of-CLEVR Image](/files/180529_rl_model/fig4.png)


#### relational network 주요 Code

- Convolution feature generation Code
```

```

- relational network Code
```

```

#### Performance


#### 추가 활용 방안

해당 방법은 다른 영역에서도 활용이 가능합니다. 만약 내가 3가지 형태의 domain 데이터가 있고 이들의 관계를 통해 어떤 결과를 알고 싶다고 하면 다음과 같은 형태로도 활용이 가능합니다.
![fig5](/files/180529_rl_model/fig5.png)

우리가 알고 싶은 영역의 데이터에 대해 object로 변환을 하고 해당 object 간의 관계를 파악하는 형태로 하여 relational network를 활용 할 수 있습니다.

#### 참조 코드
relational network를 MXNet(Gluon)을 활용하여서 구현을 해 보았습니다. 해당 코드는 다음 링크를 통해 확인하시기 바랍니다.
[Source Code](https://github.com/seujung/dl_study_with_gluon_2nd/tree/master/relational_network)
