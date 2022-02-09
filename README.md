# Purpose of repository

- 다음과 같은 5단계로 논문을 읽어나가고 담겨있는 내용과 idea를 정리한다. 
  
1. Attention + CNN
2. Transformer and Variants
3. Nature of Transformer
4. Positional Encoding of Transformer
5. Injecting bias to Transformer via structure change
6. Using Transformer in Re-ID

- 단계별로 앞으로 읽을 논문과 읽은 논문은 정리한다. 논문이 나온 시점과 다른 논문들과의 관계를 정리한다.
- 각 논문의 실험결과와 주장들을 근거로 삼아 idea를 구성 한다.   

# Short-term plan
 일주일 단위로 구성하는 단기 계획.   __읽은 논문은 bold체로 표시.__
- 02/07 ~ 02/14 Injecting bias to Transformer
   1. Visformer : The Vision-friendly Transformer (ICCV 2021, 2021/12/18) [[LINK]](http://openaccess.thecvf.com/content/ICCV2021/html/Chen_Visformer_The_Vision-Friendly_Transformer_ICCV_2021_paper.html)
   2. __ConViT: Improving Vision Transformer with Soft Convolutio nal Inductive Biases (PMLR 2021, 2021/03/19)__[[LINK]](https://arxiv.org/abs/2103.10697)
   3. CMT : Convolutional Neural Networks Meet Vision Transformers (arxiv,2021/07/13)[[LINK]](https://arxiv.org/abs/2107.06263)
   4. LocalViT : Bringing Locality to Vision Transformers (arxiv,2021/04/12)[[LINK]](https://arxiv.org/abs/2104.05707)
   5. __Swin Transformer : Hierarchical Vision Transformer using Shifted Window (ICCV 2021)__[[LINK]](https://openaccess.thecvf.com/content/ICCV2021/html/Wu_CvT_Introducing_Convolutions_to_Vision_Transformers_ICCV_2021_paper.html)
   6. CvT : Introducing Convolutions to Vision Transformers (ICCV 2021) [[LINK]](https://openaccess.thecvf.com/content/ICCV2021/html/Wu_CvT_Introducing_Convolutions_to_Vision_Transformers_ICCV_2021_paper.html)
   7. ViTAE: Vision Transformer Advanced by Exploring Intrinsic Inductive Bias(NIPS 2021) [[LINK]](https://proceedings.neurips.cc/paper/2021/hash/efb76cff97aaf057654ef2f38cd77d73-Abstract.html)
# Groups

## 개념별로 논문들을 분류한다. 
*읽은 것은 이탤릭체*  
__그 중에서도 중요한 논문은 + bold체__

1. Attention + CNN
   1. Attention augmented convolutional networks
   2. How much position information do convolutional neural networks encode?
   3. Non-local neural networks
2. Transformer and Variants
   1. *(Transformer) Attention is all you need*
   2. *(BERT) Bert: Pre-training of deep bidirectional transformers for language understanding*
   3. *__(ViT) An image is worth 16x16 words: Transformers for image recognition at scale__*
   4. (DeiT) Training data-efficient image transformers & distillation through attention
3. Nature of Transformer
   1. When Vision Transformers Outperform ResNets without Pretraining or Strong Data Augmentations
   2. *__Do vision transformers see like convolutional neural networks?__* [[LINK]](https://arxiv.org/abs/2108.08810) [[Summary]](#32-do-vision-transformers-see-like-convolutional-neural-networks)
   3. On the Expressive Power of Self-Attention Matrices
   4. (LayerNorm) Improved Robustness of Vision Transformer via PreLayerNorm in Patch Embedding
   5. (LayerNorm) On Layer Normalization in the Transformer Architecture
   6. Going deeper with Image Transformer (ICCV 2021, 2021/03/31) [[LINK]](https://arxiv.org/abs/2103.17239)
4. Positional Encoding of Transformer
   1. *__On the relationship between self-attention and convolution layers (ICLR 2020)__* [[LINK]](https://arxiv.org/abs/1911.03584)
      - Supplementary [[LINK]](http://jbcordonnier.com/posts/attention-cnn/)
   2. *Can Vision Transformer Perform Convolution? (ICLR 2022 underreview, 2021/11/02)* [[LINK]](https://arxiv.org/abs/2111.01353)
   3. *__On position embeddings in BERT(ICLR 2021, 20/09/29)__* [[LINK]](https://openreview.net/forum?id=onxoVA9FxMw)
   4. __*Rethinking positional encoding in language pre-training (ICLR 2021, 20/06/28)*__ [[LINK]](https://arxiv.org/abs/2006.15595)
   5. *Do we Really Need Explicit Position Encodings for Vision Transformers?* (21/02/22) 
      - Conditional Positional Encodings for Vision Transformers (21/03/18 revised version) [[LINK]](https://arxiv.org/abs/2102.10882)
   6. *__Rethinking and Improving Relative Position Encoding for Vision Transformer__*(ICCV 2021, 21/07/29) [[LINK]](https://arxiv.org/abs/2107.14222)
   7. *Stand-Alone self-Attention in Vision models* (NIPS 2019, 2019/06/13) [[LINK]](https://arxiv.org/abs/1906.05909)
   8. *Self-Attention with Relative Position Representations* (NAACL 2018, 2018/03/06) [[LINK]](https://arxiv.org/abs/1803.02155)
   9. *What do position embeddings Learn? An Empirical Study of Pre-Trained Language Model Positional Encoding (EMNLP 2020, 2020/09/28)* [[LINK]](https://arxiv.org/abs/2010.04903)
   10. Improve Transformer Models with Better Relative Position Embeddings(EMNLP 2020 ,20/09/28) [[LINK]](https://arxiv.org/abs/2009.13658)
5. Injecting bias to Transformer via structure change
   1. Visformer : The Vision-friendly Transformer (ICCV 2021, 2021/12/18) [[LINK]](http://openaccess.thecvf.com/content/ICCV2021/html/Chen_Visformer_The_Vision-Friendly_Transformer_ICCV_2021_paper.html)
   2. ConViT: Improving Vision Transformer with Soft Convolutio nal Inductive Biases (PMLR 2021, 2021/03/19)[[LINK]](https://arxiv.org/abs/2103.10697)
   3. CMT : Convolutional Neural Networks Meet Vision Transformers (arxiv,2021/07/13)[[LINK]](https://arxiv.org/abs/2107.06263)
   4. LocalViT : Bringing Locality to Vision Transformers (arxiv,2021/04/12)[[LINK]](https://arxiv.org/abs/2104.05707)
   5. Swin Transformer : Hierarchical Vision Transformer using Shifted Window (ICCV 2021)[[LINK]](https://openaccess.thecvf.com/content/ICCV2021/html/Wu_CvT_Introducing_Convolutions_to_Vision_Transformers_ICCV_2021_paper.html)
   6. CvT : Introducing Convolutions to Vision Transformers (ICCV 2021) [[LINK]](https://openaccess.thecvf.com/content/ICCV2021/html/Wu_CvT_Introducing_Convolutions_to_Vision_Transformers_ICCV_2021_paper.html)
   7. ViTAE: Vision Transformer Advanced by Exploring Intrinsic Inductive Bias(NIPS 2021) [[LINK]](https://proceedings.neurips.cc/paper/2021/hash/efb76cff97aaf057654ef2f38cd77d73-Abstract.html)
6. Using Transformer in Re-ID 

# Supplementary 
- Transformer
  - Illustrated Transformer [[LINK]](https://jalammar.github.io/illustrated-transformer/)
  - Illustrated ViT [[LINK]](https://medium.com/analytics-vidhya/illustrated-vision-transformers-165f4d0c3dd1)
  - Swin Transformer [[LINK]](https://www.youtube.com/watch?v=2lZvuU_IIMA)
- Positional Embedding
  - Master Positional Encoding : Part 1 [[LINK]](https://towardsdatascience.com/master-positional-encoding-part-i-63c05d90a0c3)
  - Master Positional Encoding : Part 2 [[LINK]](https://towardsdatascience.com/master-positional-encoding-part-ii-1cfc4d3e7375) 
  - AI coffee break playlist [[LINK]](https://www.youtube.com/watch?v=1biZfFLPRSY&list=PLpZBeKTZRGPOQtbCIES_0hAvwukcs-y-x)
  
# Process

각 부분에 대해 공부가 얼마나 진행되고 있는지 나타낸다. 근데 100%를 채울수 있을까...?
1. Attention + CNN 
2. About Transformer (80%)
3. Nature of Transformer (60%)
4. Positional Encoding of Transformer (70%)
5. Injecting bias to Transformer (0%)
6. Using Transformer in Re-ID (0%) 

# KEY IDEA 

여러 논문들을 읽으면서 얻은 KEY IDEA 들을 적는다. Groups의 index로 논문을 표시한다.

## 1. Attention + CNN 
## 2. About Transformer 
## 3. Nature of Transformer 
  - ## 3.2 Do Vision Transformers See Like Convolutional Neural Networks?
    - 주제
      - Vision transformer가 "무엇"을 "어떻게" 학습하는지 CNN과 비교하여 직관적으로 이해하기 좋게 visualize한 논문
    - 핵심 내용
      1. ViT는 CNN에 비해 모든 layer에서 uniform한 representation을 학습한다.
        - ![Fig1](https://user-images.githubusercontent.com/62092317/152669780-a5329c1f-a8d9-4c49-b933-a3f68315f07c.png)왼쪽 : ViT, 오른쪽 : ResNet , ViT는 모든 layer에서의 feature가 연관성이 있는 반면, ResNet은 높은 layer와 낮은 layer에서의 feature가 확실히 구분된다.
      2. ViT가 후반부에서 보는 feature는 ResNet의 초반부의 feature와 다르다.
       ![Fig2](https://user-images.githubusercontent.com/62092317/152669782-beb6e06a-49ec-4b88-b2d6-7e546f1241ef.png) 또한, ViT의 초반부 layer(0-40)의 feature와 ResNet에서의 중반부 layer(0-60)의 feature가 비슷하다.
      3. ViT는 낮은 layer에서도 global한 정보와 local한 정보에 동시에 attend 할 수 있다. 높은 layer에서는 거의 모든 attention head들이 global한 정보에 attend 한다. 그러나, train data가 충분하지 않을 경우엔 ViT의 낮은 layer가 local한 정보에 대해 attend하지 못한다.(화살표로 표시)
        - ![Fig3](https://user-images.githubusercontent.com/62092317/152669784-4012edb7-6932-46ab-ba03-8d7b0e55bcd8.png)
      4. ViT가 보는 feature를 attention head의 mean distance에 따라 ResNet의 lower layer feature과 비교해 보았을때, ViT가 attend하는 local한 정보가 ResNet의 lower layer feature와 비슷함을 알 수 있다. (2번 그림과 연결됨, Mean distance 작음: local한 정보, Mean distance 큼 : global한 정보) 
        - ![Fig4](https://user-images.githubusercontent.com/62092317/152669785-34e6ba9f-64bc-4305-af02-7ffcab264e06.png)
      5. (중요) CLS token은 초반부 MLP layer의 skip connection을 많이 통과 하며,후반부 block 에서 main branch를 통과한다. 또한, Skip connection은 CNN보다 ViT에서 훨씬 더 effective하게 동작한다. 
        - ![Fig5](https://user-images.githubusercontent.com/62092317/152669786-03ca8fa6-298d-43f4-90d4-aed157660ada.png)
          - 왼쪽 그림 
            - 0번 token은 CLS token, Ratio of Norm의 크기가 클 수록 Skip connection branch를 많이 통과한 것이고 작을 수록 main branch를 많이 통과한 것
          - 오른쪽 그림
            - ResNet과의 비교로, ViT에서 전반적으로 skip connection이 많이 사용됨을 알 수 있다.
          - CLS token은 다른 patch token들과는 완전히 반대된 경향을 보인다. 즉 CLS token은 lower layer에서 skip connection을 많이 통과하는 반면, patch token들은 higher layer에서 많이 통과한다.
        - ![Fig10](https://user-images.githubusercontent.com/62092317/152669778-3a8f388a-7584-4f23-901b-67d4978b67a2.png)
          - (NOTE) 5-1 과 5-2 의 y axis (Block index) 방향이 반대임
          - 5-1의 결과를 MLP와 SA의 skip connection 으로 나누어 분석할 때, CLS token은 SA보단 MLP layer에 많이 영향을 받음
          - Cosine similarity graph에서 값이 1에 가까울 수록 skip connection을 통과한다고 볼수 있으며, 대응되는 부분을 빨간색 화살표로 표시. CLS token의 경우 lower layer에서 MLP, SA 모두에서 skip connection을 통과하며 higher layer에서 MLP main branch에 영향을 받는 것을 알 수 있음.
      6. (중요) 기존의 ViT처럼 CLS token을 통해 linear probing을 하였을 때, ViT는 higher layer까지 token의 spatial location 정보를 잘 유지한다.
        - ![Fig6](https://user-images.githubusercontent.com/62092317/152669787-1e86650b-86d2-404a-a433-4ab9cd53425d.png)
        - ![Fig6-1](https://user-images.githubusercontent.com/62092317/152669788-af429bd5-898a-4e5e-b4a6-eaeb2ae02f29.png)
        - 그러나 모든 token들의 GAP를 사용하여 linear probing을 할 경우 , ViT 또한 각 token의 spatial location 정보를 유지하지 못하며 모든 token들이 비슷한 location 정보를 갖게 된다.
        - ![Fig7](https://user-images.githubusercontent.com/62092317/152669789-1931c803-8644-411c-8355-26323757b5a2.png)
          - 왼쪽 그림
            - 각각의 token들을 이용하여 test 한후 평균 average를 구한 것. 빨간색의 경우 CLS token으로 linear probing을 한 후 각각의 token들에 대해 성능을 측정한 후 평균을 했기 때문이다. 즉, CLS token을 제외한 나머지 token 들은 CLS에 대한 정보가 아닌 각 token들의 spatial location 정보를 지닌 token들이기 때문에 test시 성능이 높지 않다. GAP로 학습할 경우 모든 token들이 비슷한 정보를 가지며 CLS를 대표하도록 학습되기 때문에 어떤 token(CLS token도 포함)으로 성능을 측정해도 성능이 잘 나온다.
          - 오른쪽 그림
            - GAP ; First token 과 GAP ; GAP 는 결과적으론 마지막 layer에서 비슷한 성능을 내지만, First token(CLS token)의 경우 그림 5의 학습과정 때문인지 후반부 layer에서 test 성능이 급격히 증가하는 것을 알 수 있다. 반면에, GAP ; GAP는 어느 구간의 layer에서도 test 성능이 꾸준히 높다. 즉, GAP로 학습을 수행해도 그림 5에서 CLS token과 patch token의 학습 차이는 존재한다. 
            - CLS; GAP except CLS token과 CLS; GAP를 보면 전자가 미세하지만 전 구간에 대해 성능이 높게 측정됨을 알 수 있다. GAP를 사용할 때 CLS token의 역할이 다른 token들 보다 약하다는 증거로 볼 수 있다.
        - ![Fig13](https://user-images.githubusercontent.com/62092317/152740247-545a0cf7-75e9-49e7-bf43-3483d5bf743e.png)
          -  ViT를 CLS token으로 train 시킬 때, GAP로 train 시킬때에 대해서 각 token들로 classification을 했을 때 accuracy를 나타낸 것
          - 앞서 봤던 그래프에서의 결과와 동일한 양상을 보이는데(y axis 값 주의), 두 경우 모두 Layer 6까지는 비슷한 경향을 보인다는 것에 집중
          - (중요)이를 통해 학습의 차이가 classifier단과 가까운 layer에서 매우 크게 존재하고, 그 이전 layer까지는 비슷하게 학습이 진행된다고 볼 수 있다.

      7. Train data가 충분하지 않을 때, ViT는 higher layer에서 충분한 representation을 학습하지 못한다.  
      - ![Fig8](https://user-images.githubusercontent.com/62092317/152669775-ee928377-9323-41e3-bcfb-c88599e779cc.png)
        - 반면에 data의 양에 상관 없이, lower layer에서의 representation은 유지된다.
        - Intermediate representation은 data의 양에 큰 영향을 받으며, 성능에 미치는 영향이 크다. 학습 데이터가 많을 수록 ViT는 중간 layer에서 high quality의 representation을 배우게 된다.
      8. ViT는 model size, data에 상관 없이 lower layer에서 유사한 representation을 학습한다.
        - ![Fig11](https://user-images.githubusercontent.com/62092317/152669779-0680c71a-fa7d-4ffa-b7ab-7b8bb7bdf2af.png)
      9. (중요)ViT가 모든 layer에 걸쳐 uniform한 representaion을 배우고, spatial한 location을 유지할 수 있는 것은 skip connection의 영향이 크다.
        - ![Fig12](https://user-images.githubusercontent.com/62092317/152680128-d7b9182a-fce8-48db-baad-8da1574b9ce9.png)
        - 특정 Block내의 skip connection을 없앨 경우, 그 이후 block에서의 representation과 이전 block에서의 representation간 괴리가 매우 큼
        - ![Fig9](https://user-images.githubusercontent.com/62092317/152669777-d3e232a3-d6a9-4455-a992-fc698027f74a.png)
        - Receptive field 또한 이러한 영향 때문에 center patch에 dominate하게 형성된다.
    - 생각
      - __Input patch의 corresponding output patch는 각 input patch와 correlation이 가장 높다.__ 즉, CNN에선 GAP를 통해 locality 정보가 사라지는 반면, ViT에선 ClS token이 cls feature를 대표하고 나머지 output patch들은 각각의 위치에 해당하는 이미지의 정보를 담고 있다.이는 향후 detection 분야에도 유용하게 쓰일수 있다고 논문에서도 언급하고 있다.
      - ReID를 생각해보면 사람의 모습이 포함되어 있는 output patch들을 골라낼 수 있고, 이들을 pair wise로 비교해볼 수 있다면 Classification에 큰 도움이 될 것으로 예상된다.
      - Data의 size가 ViT의 intermediate representation 학습을 결정하는데, imbalance한 ReID dataset에 대해서는 어떨지 생각해보아야 한다.
   - 3.3 논문은 __ViT의 원 논문__ 인데, 나에게 Transformer 그 자체보다는 Inductive bias 라는 개념을 좀 더 확고하게 잡게 해준 논문이다.
   - 3.4 와 3.5만 읽으면 ViT의 모든 부분을 한번씩은 살펴봤다고 볼 수 있겠다. 이 부분은 Inductive bias 관점 보다는 ReID에서 사용되었던 normalization과 엮어 나가면 괜찮을 것으로 보인다. 시간이 꽤 걸리는 작업일 것이라 잠깐 미뤄둔다. 
## 4. Positional Encoding of Transformer 
  - ## 4.3 On position Embeddings in BERT
    - NLP 영역에서 BERT가 여러가지 positional embedding 방법에 따라 어떻게 학습되는지를 visualize한 논문으로, 매우 유용한 정보가 많다.__ 3.2 논문 급으로 자주 열어볼 필요가 있다. 심지어, positional embedding의 property를 3가지로 구분하여 각각에 대한 실험 결과를 나열해놓았다. 하나 주의할 점은, language에 기반하여 해석이 이루어지고 있으므로 이것이 image domain에서도 해당되는지를 숙지하며 읽어야 한다.
    - Classification 부분을 분석하는 것이 핵심이다. CLS token과 positional embedding과의 관계가 면밀히 분석되어 있다. )__CLS token은 special token인데 여기에 positional embedding이 함께 적용하는 것이 nonsense 이기 때문이다.__(추후 정리 예정)
    - 다른 task는 몰라도, classification 같은 경우 language와 image의 차이가 크지 않을것이라고 예상된다. 지금까지 읽은 논문의 자료를 모으면 가장 효율적인 positional embedding 방안을 찾을 수 있을 것 같다.
  - 4.1와 4.2 논문은 연계된 내용으로 attention mechanism이 convolution을 수행할 수 있는가?를 실험적으로 증명한다.
    - 4.1는 __pixel 단위의 attention mechanism__ 이 convolution과 동일하게 동작할 수 있도록 하는 relative position encoding parameter를 수식적으로 유도해낸다.
    - 4.2는 __patch 단위의 attention mechanism__ 이 (현재 거의 모든 vision transformer의 방식) convolution과 동일하게 동작할 수 있음을 실험적으로 증명한다. 또한, __transformer가 스스로 이를 학습해 나간다는 점에서 지금까지 CNN이 초반부 layer에 locality라는 inductive bias를 주었던 것이 classification task에 대해 올바른 inductive bias였음을 증명했다__ 는 점에서 강력한 근거로 사용할 수 있다. 
      - CNN의 convolution 연산에 따른 nature인 locality inductive bias는 network가 classification을 잘 수행할 수 있도록 돕는 올바른 bias 이다.
      - Self attention mechanism은 convolution 처럼 동작할 수 있는 capacity가 있음이 증명되었다. 그러나 내 생각은, 가능은 하지만 항상 그렇게 동작하거나 capacity가 충분하다고 보기는 어렵다.  
      - Attention mechanism이 이를 배울 수 있도록 인위적으로 설정(?)하는 key는 positional embedding(learnable parameter)에 있다.
    - 위 두 논문을 Inductive bias 관점에서 생각해보자
      - CNN의 inductive bias인 locality가 transformer에선 왜 positional embedding의 설정에 따라 발현될 수 있는가?
      - __"Convolution filter를 적용하는 것"과  "attention mechanism의 key,query dot product에서의 last term" (positional correlation matrix) 둘 다 data에 independent하다는 공통점이 있기 때문이다.__
      - 즉, CNN filter의 특성 (size, stride 등)이 data에 따라 바뀌지 않듯이 key, query dot product의 last term patch들의 position 정보간의 관계에만 의존하기 때문에 또한 그러하다.
      - __즉, network에 Inductive bias를 주기 위해선 network의 구성요소 중 input data에 independent한 위치에 주어야 한다는 것이다. 또는 그러한 요소를 network에 따로 만들어 주어야 한다. ViT에서 이를 만족하는 유일한 구성 요소가 바로 positional embedding 이다.(As far as I know, 다른 요소가 있을지는 생각해보자)__
      - __이 사실을 깨달은 것이 매우 중요하다. "막연하게 positional embedding이 locality inductive bias가 될 수 있지 않을까?" 라는 생각이 "ViT 에선 Inductive bias를 주기 위해 positional embedding을 설계해야 한다" 라는 확신으로 바뀌기 때문이다.__
      - 즉 내가 할일은 기존 CNN에서 사람들이 ReID를 잘 수행하기 위해 network에 주었던 specific한 inductive bias가 있다면 이를 찾고, 이를 positional embedding에 반영할 수 있는 방법을 찾는 것이다.  
    - 한가지 주의해야할 점이 있다. __위 두 논문은, linear probing 단계에서 둘다 GAP를 사용하여 성능을 측정하였다. 3.2 논문의 실험 결과와 연결지어 생각해보고 논리 전개의 모순이 있는지 확인해야 한다.__ 
  - 4.4 논문은 4.3 논문과 연계된 논문은 아니지만, 이어서 읽으면 큰 도움이 된다. 이 논문 역시 NLP영역을 다루고 있다.
    - 4.3에서도 다루었던 __CLS token과 다른 token들간의 relation position 문제를 다루는 가장 간단하면서도 효율적으로 보이는 방법(parameter masking)__ 을 제시한다. 읽고 나면 왜 이 생각을 안해봤지 싶다. __이 방법은 image에 대해서도 유효할 것으로 판단된다.__ 
    - Key, Query간의 내적에 의해 발생하던 content 과 postion 정보간 correlation이 의미 없다고 판단하고, 이를 없애기 위해 content와 position의 연산을 독립적으로 수행한다. 개인적으로 매우 설득력 있다고 생각한다. __특정 단어가 문장의 몇번째 위치에 와야한다는 발상 자체가 nonsense이기 때문이다.__ 나도 이 term들의 역할이 궁금했는데, 어느정도 해소되었다.
    - 그러나 image에서도 그러할까? 물론, __"특정 class를 나타내는 이미지에서 특정 position에 class의 특정 모습이 나타나야 된다."__ 라는 말 또한 nonsense 하다. 그런데, ReID dataset을 생각해보자. Query와 Gallery의 이미지는 사람의 모습을 crop한 이미지 이기 때문에(즉, 모든 ID에 대해) data의 특성이 매우 강하다고 볼 수 있다.(여러 class의 다양한 모습들이 섞여있는 dataset에 비해) __내 생각엔, language 또는 일반적인 image dataset 보다는 content와 position간 관계의 영향력이 존재할 것으로 보인다.__ 그러나, 이에 대해 바로 떠오르는 counter logic 또한 존재 한다. __ReID dataset의 고질적인 문제인, rescale된 image가 들어간다는 점이다. 따라서 어떤 image가 어떻게 rescale 됐냐에 따라서 앞서 말한 content와 position간 관계가 희미해질수 밖에 없다.__  Absolute positional embedding에 대해선 더 심각한 문제이다. 다행스럽게도, 4.6 논문이 이를 세부적으로 다루고 있다.
    - 여기서 생각의 확장이 이루어진다. Positional encoding으로 사람의 신체 부위간 position 관계를 학습하도록 유도할수 있는가? 일단 이 과정은 relative positional encoding으로 이루어 질 수 밖에 없다. 신체 부위간 상대적인 관계를 유도하는 것이 설득력 있기 때문이다. 
  - 4.5 논문은 Explicit이란 단어가 key word이다. 우리가 명시적으로 positional embedding을 위한 component를 만드는 것이 아니라, __network의 parameter들이 implicit 하게 이를 배울 수 있도록 하는 어떠한 module을 추가하자는 것__ 이 주 idea이다.
  이 module은 Transformer encoder의 출력단에 conv layer 형식으로 추가되는데, 이러한 방식의 근거는 1.2 논문을 따르는 것 같다. __Module이 추가되는 위치가 출력단이라는 것이 중요하다.__ 1.2 논문을 아직 읽어보지는 못했으나, 아마 convolution layer 자체가 position을 encoding하는 능력을 갖고 있다는 얘기가 나올것으로 보인다.(padding과 관련이 있다.) 이것은 다시 원점으로 돌아가 __"Convolution layer의 inductive bias가 positional encoding과 관련이 있다."__ 라는 주장의 근거가 된다. 모든 논문들이 같은 곳을 암묵적으로 가리키고 있는 듯한 느낌을 받는다. __이 내용은 5에 수록된 논문들과 직접적으로 연관될 것으로 보인다. 생각을 정리한 후 5로 넘어가면 많은 것을 얻을 수 있겠다.__ 
  - 4.6 논문은 앞서 언급하였듯이 absolute positional encoding(APE)이 image의 어떻게 rescale 됐냐에 매우 취약하다는 점을 지적하며 2D relative positional encoding(RPE)을 도입한 논문이다. __내가 어렴풋이 생각하고 있는 RPE의 개선 방향과 일치하긴 하는데, ReID의 경우 더 specific하게 들어갈 수 있을 것으로 보인다.__ Image Classification task에 대해서는 개선된 RPE가 APE보다 성능이 나을 수 있음을 보여주어서 확신을 가질 수 있었다. 그러나, object detection task에 대해선 APE의 성능이 더 높다고 하는데, 이는 정확한 좌표(bounding box)를 알아야하는 task의 특성상 설득력 있는  결과로 보인다. 4.3과 연결지어 읽어보자. __꽤 최근에 나온, A급 논문이 내가 원하는 방향에 대한 근거가 된다는 것이 정말 다행이다.__
## 5. Injecting bias to Transformer 
  - 5.5 Swin Transformer는 convolution 의 filter 개념을 window 라는 개념으로 치환하여 적용한 Transformer이다. 방법이 아무리 복잡하더라도, 개념적으로 쉽고 납득이 잘가면서 좋은 결과를 얻을 수 있다는 것이 매우 매력적이다. Window라는 개념이 ReID에 쓰인다면 어떻게 쓰여야 할까..? 사람의 특성을 생각하여 Vertical한 방향으로 Window를 확장해 나간다면..?
## 6. Using Transformer in Re-ID 
  - 4,5를 완벽하게 끝내면, 여기는 오히려 얼마 안걸릴 것 같다. 읽을 논문도 그리 많지 않아보인다.

# IMPORTANT FACTS
  -  Transformer를 이용한 Classification를 다루는 모든 논문의 benchmarking은 2.4(DeiT)와의 비교를 통해 이루어진다. ReID는 기본적으로 classification 이므로 참고해야할듯 하다.
  - Classification에서의 효율적인 positional encoding 방식은 relative positonal encoding이다. 이 과정에서 CLS token에 대한 고려는 반드시 이루어져야 한다. 
  -  조사해본 결과, 2D relative positional encoding의 효용성에 대해선 아직 밝혀지지 않은 것이 많다. 21/07/29에 ICCV 2021에 submit 된 4.6에서도 이를 명시적으로 언급하고 있는것으로 보아, 내가 열심히만 한다면 논리적인 무언가를 만들어 낼 여지가 많아 보인다. __아직 밝혀지지 않은 것일 뿐, 새로운 방법은 무조건 존재 할것 같다.__ 
  -  아쉽게도, 현재 성능이 좋은 ReID module 들은 아직 CNN에 의존하고 있다. ReID task가 많은 inductive bias를 요구한다는 반증이기도 하다. 연구할 여지가 많다는 점에서 좋은점일지도 모른다.
  
# TODO
 - 4.3 다시 읽고 세세히 근거 찾기
# Sketch
Idea ,Facts를 기반으로 Idea를 구상한다.

### *Patch단위의 정보를 유지하는 transformer의 특성과, 사람의 신체 구조를 학습할 수 있는 relative postional encoding을 통한 inductive bias으로 ReID의 성능을 올릴 수 있는가?*
- 사람은 멀리 있는 두 인물을 비교할 때, 어떻게 비교하는가?
  1. 옷 색깔
  2. 모습 (덩치, 키)
  3. 얼굴
- 먼저, 3.2에 근거하여 output patch를 사용하는 것이 나아보인다. 어찌되었건 각 part image에 coressponding한 정보를 담고 있기 떄문이다. 골라낼 수만 있다면, element weighted triplet hard loss 처럼 loss에 적용할 수도 있겠다.
- Relative position을 사용해야한다. 이 과정에서 positional matrix를 어떻게 modeling 하느냐가 관건이 되겠다.
- 그러나, Transformer 구조에서 postional embedding의 설계만으로 이를 따라갈 수 있을것이라고는 생각하지 않는다. Additional한 conv module이 필요할 것이다.
- Module이 추가된다면, 기존 transformer가 갖고 있는 nature와 redundant한 동작을 하지 않는 역할을 할 수 있는 위치에 추가되어야 한다. 
# Implemetation
구상한 Idea를 구현해보고 결과를 확인한다.
