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
- 04/14 - 04/21
  1. T. Nguyen, M. Raghu, and S. Kornblith, "Do wide and deep networks learn the same things? uncovering how neural network representations vary with width and depth," arXiv preprint arXiv:2010.15327, 2020.
  2. Weight & Biase 
  3. 정리 다시 시작하자...

  
## 개념별로 논문들을 분류한다. 
__읽은 것은 bold체__ 
__그 중에서도 중요한 논문은 + *이탤릭체*__
1. Attention + CNN
    1. How much position information do convolutional neural networks encode?
    2. Non-local neural networks
2. Transformer and Variants
    1. __(Transformer) Attention is all you need__
    2. __(BERT) Bert: Pre-training of deep bidirectional transformers for language understanding__
    3. *__(ViT) An image is worth 16x16 words: Transformers for image recognition at scale__*
    4. (DeiT) Training data-efficient image transformers & distillation through attention
3. Nature of Transformer
   1. When Vision Transformers Outperform ResNets without Pretraining or Strong Data Augmentations
   2. *__Do vision transformers see like convolutional neural networks?__* [[LINK]](https://arxiv.org/abs/2108.08810) [[Summary]](#32-do-vision-transformers-see-like-convolutional-neural-networks)
   3. *__How Do vision Transformers Work?(ICLR,2022)__*[[LINK]](https://arxiv.org/abs/2202.06709) [[CODE]](https://github.com/xxxnell/how-do-vits-work)
   4. On the Expressive Power of Self-Attention Matrices
   5. (LayerNorm) Improved Robustness of Vision Transformer via PreLayerNorm in Patch Embedding
   6. (LayerNorm) On Layer Normalization in the Transformer Architecture
   7. Going deeper with Image Transformer (ICCV 2021, 2021/03/31) [[LINK]](https://arxiv.org/abs/2103.17239)
   8. Rethinking Spatial Dimensions of Vision Transformers (ICCV 2021 ) [[LINK]](https://openaccess.thecvf.com/content/ICCV2021/html/Heo_Rethinking_Spatial_Dimensions_of_Vision_Transformers_ICCV_2021_paper.html)
   9. UnderStanding Robustness of Transformers for Image Classification(ICCV 2021) [[LINK]](https://openaccess.thecvf.com/content/ICCV2021/html/Bhojanapalli_Understanding_Robustness_of_Transformers_for_Image_Classification_ICCV_2021_paper.html)
   10. Towards robust Vision Transformer(arxiv, 2021/05/26) [[LINK]](https://arxiv.org/abs/2105.07926)
   11. Intriguing properties of vision transformers)(NIPS 2021, 2021/11/25) [[LINK]](https://arxiv.org/abs/2105.10497)
   12. Do wide and deep networks learn the same things? Uncovering how neural network representations vary with width and depth(ICLR 2021) [[LINK]](https://arxiv.org/abs/2010.15327)
   13. __Blurs behave like ensembles : Spatial smoothings to imporve accuracy, uncertainty, and robustness__ (arxiv, 2021/11/23) [[LINK]](https://openreview.net/forum?id=34mWBCWMxh9)
   14. Vision Transformer are Robust Learners (AAAI 2022, 2021/12/4) [[LINK]](https://arxiv.org/abs/2105.07581)
   15. On the adversarial robustness of visual transformers (arxiv, 2021)[[LINK]](https://arxiv.org/abs/2103.15670)
   16. Are convolutional neural networks or transformers more like human vision?(CogSci, 2021)[[LINK]](https://arxiv.org/abs/2105.07197)
   17. Early convolutions help transformers see better(NIPS, 2021)[[LINK]](https://arxiv.org/abs/2106.14881#:~:text=Vision%20transformer%20(ViT)%20models%20exhibit%20substandard%20optimizability.&text=Using%20a%20convolutional%20stem%20in,while%20maintaining%20flops%20and%20runtime.)
   18. Pyhessian : Neural networks through the lens of the hessian (IEEE international conference on big data,2020) [[LINK]](https://ieeexplore.ieee.org/abstract/document/9378171)
   19. Incorporating Convolution Designs into Visual Transformers (ICCV 2021)[[LINK]](https://openaccess.thecvf.com/content/ICCV2021/html/Yuan_Incorporating_Convolution_Designs_Into_Visual_Transformers_ICCV_2021_paper.html)
   20. Making Convolutional Networks Shift-Invariant Again(PMLR 2019)[[LINK]](http://proceedings.mlr.press/v97/zhang19a.html)
4. Positional Encoding of Transformer
   1. *__On the relationship between self-attention and convolution layers (ICLR 2020)__* [[LINK]](https://arxiv.org/abs/1911.03584)
      - Supplementary [[LINK]](http://jbcordonnier.com/posts/attention-cnn/)
   2. __Can Vision Transformer Perform Convolution? (ICLR 2022 underreview, 2021/11/02)__ [[LINK]](https://arxiv.org/abs/2111.01353)
   3. *__On position embeddings in BERT(ICLR 2021, 20/09/29)__* [[LINK]](https://openreview.net/forum?id=onxoVA9FxMw)
   4. __Rethinking positional encoding in language pre-training (ICLR 2021, 20/06/28)__ [[LINK]](https://arxiv.org/abs/2006.15595)
   5. __Do we Really Need Explicit Position Encodings for Vision Transformers?__(21/02/22) 
      - __Conditional Positional Encodings for Vision Transformers (21/03/18 revised version)__ [[LINK]](https://arxiv.org/abs/2102.10882)
   6. __Rethinking and Improving Relative Position Encoding for Vision Transformer__(ICCV 2021, 21/07/29) [[LINK]](https://arxiv.org/abs/2107.14222)
   7. __Stand-Alone self-Attention in Vision models__(NIPS 2019, 2019/06/13) [[LINK]](https://arxiv.org/abs/1906.05909)
   8. __Self-Attention with Relative Position Representations__(NAACL 2018, 2018/03/06) [[LINK]](https://arxiv.org/abs/1803.02155)
   9. __What do position embeddings Learn? An Empirical Study of Pre-Trained Language Model Positional Encoding (EMNLP 2020, 2020/09/28)__[[LINK]](https://arxiv.org/abs/2010.04903)
   10. Improve Transformer Models with Better Relative Position Embeddings(EMNLP 2020 ,20/09/28) [[LINK]](https://arxiv.org/abs/2009.13658)
   11. Position information in transformers : An overview (arxiv, 2021/02/22) [[LINK]](https://arxiv.org/abs/2102.11090)
5. Injecting bias to Transformer via structure change
   1. Visformer : The Vision-friendly Transformer (ICCV 2021, 2021/12/18) [[LINK]](http://openaccess.thecvf.com/content/ICCV2021/html/Chen_Visformer_The_Vision-Friendly_Transformer_ICCV_2021_paper.html)
   2. ConViT: Improving Vision Transformer with Soft Convolutio nal Inductive Biases (PMLR 2021, 2021/03/19)[[LINK]](https://arxiv.org/abs/2103.10697)
   3. __CMT : Convolutional Neural Networks Meet Vision Transformers__ (arxiv,2021/07/13)[[LINK]](https://arxiv.org/abs/2107.06263)
   4. LocalViT : Bringing Locality to Vision Transformers (arxiv,2021/04/12)[[LINK]](https://arxiv.org/abs/2104.05707)
   5. *__Swin Transformer : Hierarchical Vision Transformer using Shifted Window (ICCV 2021)__*[[LINK]](https://openaccess.thecvf.com/content/ICCV2021/html/Wu_CvT_Introducing_Convolutions_to_Vision_Transformers_ICCV_2021_paper.html) [[CODE]](https://github.com/microsoft/Swin-Transformer/blob/5d2aede42b4b12cb0e7a2448b58820aeda604426/models/swin_transformer.py#L89)
   6. CvT : Introducing Convolutions to Vision Transformers (ICCV 2021) [[LINK]](https://openaccess.thecvf.com/content/ICCV2021/html/Wu_CvT_Introducing_Convolutions_to_Vision_Transformers_ICCV_2021_paper.html)
   7. ViTAE: Vision Transformer Advanced by Exploring Intrinsic Inductive Bias(NIPS 2021) [[LINK]](https://proceedings.neurips.cc/paper/2021/hash/efb76cff97aaf057654ef2f38cd77d73-Abstract.html)
6. Using Transformer in Re-ID 
   1. *__TransMatcher : Deep Image Matching Through Transfomers for Generalizable Person Re-identification(NIPS 2021)__* [[LINK]](https://papers.nips.cc/paper/2021/hash/0f49c89d1e7298bb9930789c8ed59d48-Abstract.html) [[CODE]](https://github.com/ShengcaiLiao/QAConv/tree/master/projects/transmatcher)
   2. __Person Re-identification with a Locally Aware Transformer(NIPS 2021 submit, 2021/06)__[[LINK]](https://arxiv.org/abs/2106.03720)
    - 현재 이 논문은 방법만을 제시하고 있으나, evaluation code마저 잘못짜여져 있어 성능이 충분히 의심가는 상황 (설득력 X)
   3. __TransReID : Transformer-based Object Re-Identification (ICCV 2021, 2021/03/26)[[LINK]](https://arxiv.org/abs/2102.04378)__ [[CODE]](https://github.com/damo-cv/TransReID)
   4. Self-Supervised Pre-training for Transformer-Based Person Re-identification (arxiv, 2021/11/23) [[LINK]](https://arxiv.org/abs/2111.12084) [[CODE]](https://github.com/damo-cv/TransReID)
   5. Diverse Part Discovery: Occluded Person Re-Identification With Part-Aware Transformer (CVPR 2021) [[LINK]](https://openaccess.thecvf.com/content/CVPR2021/html/Li_Diverse_Part_Discovery_Occluded_Person_Re-Identification_With_Part-Aware_Transformer_CVPR_2021_paper.html)
7. ReID (Hao Luo paper list [[LINK]](ttps://scholar.google.com/citations?user=7QvWnzMAAAAJ&hl=zh-CN))
   1. __(PCB)Beyond Part Models: Person Retrieval with Refined Part Pooling (and A Strong Convolutional Baseline)__ (ECCV 2018) [[LINK]](https://openaccess.thecvf.com/content_ECCV_2018/html/Yifan_Sun_Beyond_Part_Models_ECCV_2018_paper.html)
    [[CODE]](https://github.com/syfafterzy/PCB_RPP_for_reID)
   2. __A strong Baseline and Batch Normalization Neck for Deep Person Re-Identification__ [[LINK]](https://ieeexplore.ieee.org/abstract/document/8930088)
   3. __Bag of Tricks and A strong baseline for deep person re-identification(CVPR 2019)__ [[LINK]](https://arxiv.org/abs/1903.07071)
   4. Unsupervised Pre-training for Person Re-identification (CVPR 2021) [[LINK]](https://openaccess.thecvf.com/content/CVPR2021/html/Fu_Unsupervised_Pre-Training_for_Person_Re-Identification_CVPR_2021_paper.html)
8. Metric learning
   1. __L2-constrained softmax loss for discriminative face verification__ (arxiv 2017)[[LINK]](https://arxiv.org/abs/1703.09507)
   2. __Rethinking Feature Discrimination and Polymerization for Large-scale Recognition__(arxiv 2017)[[LINK]](https://arxiv.org/abs/1710.00870)
9. Visualization
   1. (Attention Roll out) __Quantifying Attention Flow in Transformers(arxiv,2020/05/31)__[[LINK]](https://arxiv.org/abs/2005.00928)
   2. (Grad Attention Roll out)Transformer Interpretability Beyond Attention Visualization(CVPR 2021) [[LINK]](https://openaccess.thecvf.com/content/CVPR2021/papers/Chefer_Transformer_Interpretability_Beyond_Attention_Visualization_CVPR_2021_paper.pdf)
   3. TS-CAM: Token Semantic Coupled Attention Map for Weakly Supervised Object Localization(ICCV 2021)[[LINK]](https://openaccess.thecvf.com/content/ICCV2021/html/Gao_TS-CAM_Token_Semantic_Coupled_Attention_Map_for_Weakly_Supervised_Object_ICCV_2021_paper.html)

# Supplementary 
- Transformer
  - Illustrated Transformer [[LINK]](https://jalammar.github.io/illustrated-transformer/)
  - Illustrated ViT [[LINK]](https://medium.com/analytics-vidhya/illustrated-vision-transformers-165f4d0c3dd1)
  - Swin Transformer [[LINK]](https://www.youtube.com/watch?v=2lZvuU_IIMA)
- Positional Embedding
  - Master Positional Encoding : Part 1 [[LINK]](https://towardsdatascience.com/master-positional-encoding-part-i-63c05d90a0c3)
  - Master Positional Encoding : Part 2 [[LINK]](https://towardsdatascience.com/master-positional-encoding-part-ii-1cfc4d3e7375) 
  - AI coffee break playlist [[LINK]](https://www.youtube.com/watch?v=1biZfFLPRSY&list=PLpZBeKTZRGPOQtbCIES_0hAvwukcs-y-x)
  - Visual Guide to Transformer Neural Networks - (Episode 1) Position Embeddings [[LINK]](https://www.youtube.com/watch?v=dichIcUZfOw&t=606s)
- ReID baseline Code [[LINK]](https://github.com/layumi/Person_reID_baseline_pytorch)
- How do Vision Transformers work?[[LINK]](https://www.youtube.com/watch?v=dOwRXpSSc8E)
- Visualizing Attention Map In transfomer
  - Exploring Explanability for Vision Transformers[[LINK]](https://jacobgil.github.io/deeplearning/vision-transformer-explainability) [[CODE]](https://github.com/jacobgil/vit-explain)
  - ipynb implementation[[CODE]](https://github.com/jeonsworld/ViT-pytorch/blob/main/visualize_attention_map.ipynb)
  - Transformer Interpretability Beyond Attention Visualization [[CODE]](https://github.com/hila-chefer/Transformer-Explainability)
- Weight & Biase [[LINK]](https://89douner.tistory.com/313)
  - Loss plot per epoch & iteration [[LINK]](https://github.com/wandb/client/issues/1591)

# Process

각 부분에 대해 공부가 얼마나 진행되고 있는지 나타낸다. 근데 100%를 채울수 있을까...?
1. Attention + CNN 
2. About Transformer (100%)
3. Nature of Transformer (70%)
4. Positional Encoding of Transformer (100%)
5. Injecting bias to Transformer (70%)
6. Using Transformer in Re-ID (80%)
7. ReID (90%) 

# KEY IDEA 

여러 논문들을 읽으면서 얻은 KEY IDEA 들을 적는다. Groups의 index로 논문을 표시한다.

## 1. Attention + CNN 
## 2. About Transformer 
## 3. Nature of Transformer  
  - ### 3.2 Do vision Transformers see like convolutional neural networks? [[Summary Link]](https://github.com/SeongSuKim95/WORK/blob/master/Summary/Do%20Vision%20Transformers%20See%20Like%20Convolutional%20Neural%20Networks.md)
## 4. Positional Encoding of Transformer 
  - ### 4.1 __On the relationship between self-attention and convolution layers (ICLR 2020)__
  - ### 4.2 __Can Vision Transformer Perform Convolution? (ICLR 2022 underreview, 2021/11/02)__
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
  - ### 4.3 __On position embeddings in BERT(ICLR 2021, 20/09/29)__ [[Summary Link]]()
  - ### 4.4  __Rethinking positional encoding in language pre-training (ICLR 2021, 20/06/28)__ [[Summary Link]]()
  - ### 4.5 __Do we Really Need Explicit Position Encodings for Vision Transformers?(21/02/22)__[[Summary Link]]()
  - ### 4.6 __Rethinking and Improving Relative Position Encoding for Vision Transformer__ (ICCV 2021, 21/07/29)[[Summary Link]]()

## 5. Injecting bias to Transformer 
  - 5.5 Swin Transformer는 convolution 의 filter 개념을 window 라는 개념으로 치환하여 적용한 Transformer이다. 방법이 아무리 복잡하더라도, 개념적으로 쉽고 납득이 잘가면서 좋은 결과를 얻을 수 있다는 것이 매우 매력적이다. Window라는 개념이 ReID에 쓰인다면 어떻게 쓰여야 할까..? 사람의 특성을 생각하여 Vertical한 방향으로 Window를 확장해 나간다면..?
## 6. Using Transformer in Re-ID 


# IMPORTANT FACTS
  -  Transformer를 이용한 Classification를 다루는 모든 논문의 benchmarking은 2.4(DeiT)와의 비교를 통해 이루어진다. ReID는 기본적으로 classification 이므로 참고해야할듯 하다.
  - Classification에서의 효율적인 positional encoding 방식은 relative positonal encoding이다. 이 과정에서 CLS token에 대한 고려는 반드시 이루어져야 한다. 
  -  조사해본 결과, 2D relative positional encoding의 효용성에 대해선 아직 밝혀지지 않은 것이 많다. 21/07/29에 ICCV 2021에 submit 된 4.6에서도 이를 명시적으로 언급하고 있는것으로 보아, 내가 열심히만 한다면 논리적인 무언가를 만들어 낼 여지가 많아 보인다. __아직 밝혀지지 않은 것일 뿐, 새로운 방법은 무조건 존재 할것 같다.__ 
  -  아쉽게도, 현재 성능이 좋은 ReID module 들은 아직 CNN에 의존하고 있다. ReID task가 많은 inductive bias를 요구한다는 반증이기도 하다. 연구할 여지가 많다는 점에서 좋은점일지도 모른다.
  
# Sketch
Idea ,Facts를 기반으로 Idea를 구상한다.

### *Patch단위의 정보를 유지하는 transformer의 특성과, 사람의 신체 구조를 학습할 수 있는 relative postional encoding을 통한 inductive bias으로 ReID의 성능을 올릴 수 있는가?*
- 먼저, 3.2에 근거하여 output patch를 사용하는 것이 나아보인다. 어찌되었건 각 part image에 coressponding한 정보를 담고 있기 Eo문이다. 골라낼 수만 있다면, element weighted triplet hard loss 처럼 loss에 적용할 수도 있겠다.
  - 6.2 논문이 이 Idea를 기반으로 ReID를 수행하였으나, 방법만 제시하고 해석은 제시되어 있지 않다.
- Relative position을 사용해야한다. 이 과정에서 positional matrix를 어떻게 modeling 하느냐가 관건이 되겠다.
  - Swin Transformer의 Relative postional embedding의 구현 code를 이해하면 수월하다.
- 그러나, Transformer 구조에서 postional embedding의 설계만으로 이를 따라갈 수 있을것이라고는 생각하지 않는다. Additional한 conv module이 필요할 것이다.
- Module이 추가된다면, 기존 transformer가 갖고 있는 nature와 redundant한 동작을 하지 않는 역할을 할 수 있는 위치에 추가되어야 한다. 
### *Vision Transformer의 Self attention mechanism을 metric learning에 더 유용하도록 적용할 수 있는가?* 

# Implemetation
   - 0305 : Query image에 대한 Top 10 Rank gallery visualization 완료
   - 0306 : Query image에 대한 Attention roll out 완료 
   - 0310 : Query image에 대한 Visualize 결과 통합
     ![show](https://user-images.githubusercontent.com/62092317/157599820-cb30c46e-e4b0-4a95-9584-fa64866b0327.png)
   - 0315 :  
     - Training / Weight & Bias 를 통해 attention map 연동 완료
     - HARDEST QUERY 출력 완료
     - Positional Embedding visualize 완료
   - 0322 :
     - Patch wise Triplet loss 구현
       - Cosine distance FIX 완료
       - Euclidean distance 구현 완료
     - Self - Attention 과 metric learning의 연결고리..?
       - 마지막 transformer layer에 anchor, negative, positive 간 self-attention이 고려된 부분이 추가된다면?
  2. Weight & Bias 의 분석 tool 사용법을 완전히 익힌다.

  3. Matplotlib 사용법
   - Subplot 기본 [[LINK]](https://soooprmx.com/matplotlib%EC%9D%98-%EA%B8%B0%EB%B3%B8-%EC%82%AC%EC%9A%A9%EB%B2%95-%EB%B0%8F-%EB%8B%A4%EB%A5%B8-%EC%8B%9C%EA%B0%81%ED%99%94-%EB%9D%BC%EC%9D%B4%EB%B8%8C%EB%9F%AC%EB%A6%AC/)
   - Subplot 간격 설정 [[LINK]](https://steadiness-193.tistory.com/174)
   - cv2 Attention map in PLT [[LINK]](http://www.learningaboutelectronics.com/Articles/How-to-display-an-OpenCV-image-in-Python-with-matplotlib.php)

  4. Dealing with Pytorch Model params [[LINK]](https://comlini8-8.tistory.com/50) [[LINK]](https://tutorials.pytorch.kr/beginner/saving_loading_models.html)

  5. Torch.Detach [[LINK]](https://redstarhong.tistory.com/64)
  
  6. TSNE for Debug [[LINK]](https://learnopencv.com/t-sne-for-feature-visualization/)
 구상한 Idea를 구현해보고 결과를 확인한다.

