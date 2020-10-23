# JinYeJin.github.io

# 논문읽기 두번째 :: [DeepLab V3] Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation

 

[TOC]



##  들어가기 전에

[발표자료? 추가할지말지 미정](https://docs.google.com/presentation/d/1kzSL0EqgfBKvNl4MA2LaVsKBKL3i9qxD5pO8BeiWnIc/edit?usp=sharing)



<iframe width="1189" height="669" src="https://www.youtube.com/embed/ATlcEDSPWXY" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>



### 제목소개

Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation

- Semantic - 의미론적의란 뜻인데 이게 세그멘테이션에서 어떤 것을 의미하는지 자료 찾아서 넣기
- Segmentation
- 왜 segmentation 문제를 해결하는 데에 Encoder-Decoder를 사용하였나?
- 왜 제목에  Atrous Separable Convolution가 들어가 있는가?
- 



### 논문읽기 가이드 라인



| 공통질문                                     | 예시 (LSTM)                                                  |
| -------------------------------------------- | ------------------------------------------------------------ |
| 저자는 어떠한 문제를 해결하고자 했는가?      | 이전의 방법에 새로운 방식을 추가하여 각 방법의 장점을 취해 세그멘테이션 문제를 해결하려 하였다. (이전의 주류 방식이 뭐였는지 간단 조사) |
| 그 문제를 해결한 방법은 무엇인가?            | 새로운 decoder를 추가?                                       |
| 그 방법에 대한 이해 without 수학 (intuition) |                                                              |
| 그 방법에 대한 이해 with 수학                |                                                              |
| 성공적으로 목표를 달성했나?                  | PASCAL VOC  012 과 Cityscapes datasets에서 테스트 셋에서 어떠한 전처리도 없이 퍼포먼스를 89.0% 과 82.1%로 효율성을 입증했다. |
| 부족한 부분에는 무엇이 있나?                 |                                                              |
| 논문을 코드로 구현해보는 간단한 예제?        |                                                              |



### DeepLab의 배경

deeplab v1에서 내용 발췌?



### DeepLab 버전 소개

#### 각 논문의 제목

- DeepLab V1
  Semantic Image Segmentation with Deep Convolutional Nets and Fully Connected CRFs. ICLR 2015.
- DeepLab V2
  DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs. TPAMI 2017.
- DeepLab V3
  Rethinking Atrous Convolution for Semantic Image Segmentation. arXiv 2017.

- DeepLab V3+
  Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation. arXiv 2018.

#### 비교

| Version                   | v1(2015)                  | v2(2017)                                 | v3(2017)                                                     | v3+(2018)                                                    |
| ------------------------- | ------------------------- | ---------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 기본 모듈                 |                           |                                          | ASPP                                                         | ASPP                                                         |
| Backbone                  |                           |                                          |                                                              | Xception? ResNet-101?                                        |
| 이전버전과 달리 추가된 점 | atrous convolution을 적용 | Atrous Spatial Pyramid Pooling 기법 제안 | 기존 [ResNet](https://arxiv.org/abs/1512.03385) 구조에 atrous convolution을 활용해 좀 더 dense한 feature map을 얻는 방법을 제안 | separable convolution과 atrous convolution을 결합한 atrous separable convolution의 활용을 제안 |
| 이전버전과 달리 삭제된 점 |                           |                                          |                                                              |                                                              |







## Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation



### Abstract

Abstract. Spatial pyramid pooling module or encode-decoder structure are used in deep neural networks for semantic segmentation task. The former networks are able to encode multi-scale contextual information by probing the incoming features with filters or pooling operations at multiple rates and multiple effective fields-of-view, while the latter networks can capture sharper object boundaries by gradually recovering the spatial information. In this work, we propose to combine the advantages from both methods. Specifically, our proposed model, DeepLabv3+, extends DeepLabv3 by adding a simple yet effective decoder module to refine the segmentation results especially along object boundaries. We further explore the Xception model and apply the depthwise separable convolution to both Atrous Spatial Pyramid Pooling and decoder modules, resulting in a faster and stronger encoder-decoder network. We demonstrate the effectiveness of the proposed model on PASCAL VOC  012 and Cityscapes datasets, achieving the test set performance of 89.0% and 82.1% without any post-processing. Our paper is accompanied with a publicly available reference implementation of the proposed models in Tensorflow at https://github.com/tensorflow/models/tree/master/research/deeplab.

Keywords: Semantic image segmentation, spatial pyramid pooling, encoderdecoder, and depthwise separable convolution.



Spatial pyramid pooling module 이나 encode-decoder structure은 시멘틱 세그멘테이션에서 깊은 신경망에서 사용된다. 이전의 네트워크들은 다양한 스케일의 여러 rates 와 여러 effective fields-of-view에서 필터링되거나 풀링작업을 수행한 주어진 feature [contextual information](https://en.wikipedia.org/wiki/Contextual_image_classification)을 인코딩할 수 있는 반면 최근의 네트워크들은 공간 정보를 점차 복수하면서 더 샤프한 체의 경계를 잡아낼 수 있다. 이 연구에선, 두 방범의 장점만을 취했다. 특히, 우리가 보여주는 DeepLabv3+ 모델은 효과적으로 물체의 경계를 세그멘테이션 결과를 세분화하기 위해서 DeepLabv3에 간단한데도 효과적인 디코더 모듈을 추가해서 확장했다. 더 나아가 더 빠르고 강력한 encoder-decoder network를 결과로 내기 위해서 Xception 모델을 탐색하고 [depthwise separable convolution](https://m.blog.naver.com/PostView.nhn?blogId=chacagea&logNo=221582912200&categoryNo=32&proxyReferer=https:%2F%2Fwww.google.com%2F)을 Atrous Spatial Pyramid Pooling 과 decoder modules에 적용한다. 이 모델로  PASCAL VOC  012 과 Cityscapes datasets에서 테스트 셋에서 어떠한 전처리도 없이 퍼포먼스를 89.0% 과 82.1%로 효율성을 입증했다. 



키워드

- Semantic image segmentation
- spatial pyramid pooling
- encoderdecoder
- depthwise separable convolution.





### 1 Introduction

– We propose a novel encoder-decoder structure which employs DeepLabv3 as a powerful encoder module and a simple yet effective decoder module. 

– In our structure, one can arbitrarily control the resolution of extracted encoder features by atrous convolution to trade-off precision and runtime, which is not possible with existing encoder-decoder models.

– We adapt the Xception model for the segmentation task and apply depthwise separable convolution to both ASPP module and decoder module, resulting in a faster and stronger encoder-decoder network.

– Our proposed model attains a new state-of-art performance on PASCAL VOC 2012 and Cityscapes datasets. We also provide detailed analysis of design choices and model variants.

– We make our Tensorflow-based implementation of the proposed model publicly available at https://github.com/tensorflow/models/tree/master/research/deeplab.



- DeepLabv3를 강력한 인코더 모듈로 사용한 새로운 encoder-decoder 구조와 간단하지만 효과적인 디코더 모듈을 선보인다.
- 이 구조는 실행시간과 precision(정밀도, TP / (TP + FP))를 trade-off하기위해서 atrous convolution으로 추출된 인코더의 featrue들을 해상도를 임의로 조작할 수 있다.
- Xception 모델을 세그멘테이션 작업을 위해서 변형했고, 더 빠르고 강력한 enxoder-decoder 네트워크를 위해 depthwise separable convolution 을 ASPP module과 decoder module 둘 모두에 적용했다.
- 이 모델은  PASCAL VOC 2012 and Cityscapes datasets에서 새로운 SOTA 성능에 도달했다. 그리고 디자인 선택과 다양한 모델의 세부적인 분석을 제공한다.
- https://github.com/tensorflow/models/tree/master/research/deeplab에서Tensorflow를 기반으로 구현한 모델을 제공. 





### 2 Realated Work

기본적으로 모델은 Fully Convolutional Networks(FCNs)를 기반으로 한다. 

#### Spatial pyramid pooling

PSPNet, DeepLab과 같은 모델이 다양한 grid scale(욜로에서 설명하는 그 그리드인가?) 에서 Spatial pyramid pooling이나 몇몇의 각기 비율이 다른 parallel atrous convolution 으로 동작한다.

PSPnet은 SPPnet을 기반으로 만든 것인가??

PSPnet의 논문의 참조 12번이 SPPnet이다. Pyramid Scene Parsing Network

그럼 ASPP는 언제부터 적용? 구글이 직접 고안해낸 방법?

#### Encoder-decoder

알짜 딥러닝 14장(자료 첨부)

Encoder-decoder 구조는 human pose estimation, object detection 그리고 semantic segmentation 등 여러 분야에 활용된다.

(1) Encoder 모듈은 피처맵을 점진적으로 줄여나가고, 더 상위의 semantic information을 잡아낸다.

(2) Decoder 모듈은 spatial infromation을 점진적으로 되돌린다.

위와 같은 Encoder-decoder의 특성을 이용해서 DeepLabv3를 만듬, simple yet effective라는 구절을 아주 좋아하는 모앙.

[Fig. 2. 넣기]

#### Depthwise separable convolution

Depthwise separable convolution이나 group convolution은 퍼포먼스는 유지하거나 약간 더 좋아지고, 컴퓨팅 비용과 파리미터를 줄이는 강력한 연산이다. Depthwise separable convolution가 여러 모델에 사용되지만, 이 논문에선 특히 Xception을 사용했다. 왜?? COCO 2017 데이터셋에서 성능 향상을 보여서?





### 3 Methods

Atrous convolution과 depthwise separable convolution을 간단설명한다. DeepLabv3에 대한 간단리뷰. 왜? DeepLabv3+의 인코더 모듈로 쓰여서. 그 담에 인코더에 붙일 디코더도 설명할꺼다. 글고 수정된 Xception 모듈도 보여줄께~이거 짱짱맨~짱좋아짐~ 

#### 3.1 Encoder-Decoder with Atrous Convolution

##### Atrous convoultion

[Fig. 3.]

- <img src="https://latex.codecogs.com/gif.latex?O_t=\text { Onset event at time bin } t " /> 
  - <img src="https://latex.codecogs.com/gif.latex?O_t=\text { Onset event at time bin } t " /> - <img src="https://latex.codecogs.com/gif.latex?O_t=\text { Onset event at time bin } t " /> 

- <img src="https://latex.codecogs.com/gif.latex?O_t=\text { Onset event at time bin } t " /> 





- <img src="https://latex.codecogs.com/gif.latex?O_t=\text { Onset event at time bin } t " />

- <img src="https://latex.codecogs.com/gif.latex?y[i] = \sum_{k}x[i+r\cdot k]w[k] />
- 
- 

```latex
- <img src="https://latex.codecogs.com/gif.latex?O_t=\text { Onset event at time bin } t " />
- <img src="https://latex.codecogs.com/gif.latex?y[i] = \sum_{k}x[i+r\cdot k]w[k] />
```

<script type="text/javascript"
        src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.0/MathJax.js?config=TeX-AMS_CHTML"></script>

$\text

- <img src="https://latex.codecogs.com/gif.latex?y[i] = \sum_{k}x[i+r\cdot k]w[k]$ />

- <img

- <img src="https://latex.codecogs.com/gif.latex?y[i] = \sum_{k}x[i+r\cdot k]w[k]$ />

- 

```math
y[i] = \sum_{k}x[i+r\cdot k]w[k]
```

$`y[i] = \sum_{k}x[i+r\cdot k]w[k]`$



#### Modified Aligned Xception



### 4 Experimental Evaluation

#### Decoder Design Choices

#### ResNet-101 as Network Backbone

#### Xception as Network Backbone

Xceptino은 Inception 모델의 발전형?

알짜딥러닝 9장에 Resnet과 Inception 설명이 있으니 복습하면서 보충설명

#### Improvement along Object Boundaries

#### Experimental Results on Cityscapes



### 5 Conclusion

Our proposed model “DeepLabv3+” employs the encoder-decoder structure where DeepLabv3 is used to encode the rich contextual information and a simple yet effective decoder module is adopted to recover the object boundaries. One could also apply the atrous convolution to extract the encoder features at an arbitrary resolution, depending on the available computation resources. We also explore the Xception model and atrous separable convolution to make the proposed model faster and stronger. Finally, our experimental results show that the proposed model sets a new state-of-the-art performance on PASCAL VOC 2012 and Cityscapes datasets. Acknowledgments We would like to acknowledge the valuable discussions with Haozhi Qi and Jifeng Dai about Aligned Xception, the feedback from Chen Sun, and the support from Google Mobile Vision team.



“DeepLabv3+” 은 DeepLabv3에서 rich contextual information을 인코딩하는데 사용된  encoder-decoder 구조와 오브젝트의 바운더리를 복수하는데 채택된 단순하지만 효과적인 decoder 모듈을 사용하였다. 또한 arbitrary resolution에서 encoder features를 추출하기 위해서 atrous convolution을 적용하였고, depending on the available computation resources.? 또한 propose한 모델을 빠르고 강력하게 만들기 위해서 Xception 모델과 atrous separable convolution을 분석했다. 마지막으로, 이 실험 결과는 발표한 모델이  PASCAL VOC 2012 and Cityscapes datasets에서 SOTA인 것을 보여준다. 도움 준 님들 감사. (Abstract와 똑같은 내용, 논문은 수미상관으로 적어야 한다는 것을 앎.)





## References

[DeepLabv1 논문(arXiv)](https://arxiv.org/abs/1412.7062)

[DeepLabv2 논문(arXiv)](https://arxiv.org/abs/1606.00915)

[DeepLabv3 논문(arXiv)](https://arxiv.org/abs/1706.05587)

[DeepLabv3+ 논문(arXiv)](https://arxiv.org/abs/1802.02611)

[Lunit Tech Blog - DeepLab V3+: Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation](https://blog.lunit.io/2018/07/02/deeplab-v3-encoder-decoder-with-atrous-separable-convolution-for-semantic-image-segmentation/)

논문의 참조 논문들





[Tensorflow official Code](https://github.com/tensorflow/models)

[Deeplab Demo(Google Colab)](https://colab.research.google.com/github/tensorflow/models/blob/master/research/deeplab/deeplab_demo.ipynb#scrollTo=aUbVoHScTJYe)
