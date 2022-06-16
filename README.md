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
- 4/21 - 06/11
  - 아이디어 구현 및 결과 확인 완료
  - 한글 논문 작성 완료
- 06/11 - 06/17 (기말 기간)
  - 7월 중으로 영어 논문 작성
  - 논리 전개 정리
- 06/17 - 06/24
  - 영어 논문 작성
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
   2. *__Do vision transformers see like convolutional neural networks?__* [[LINK]](https://arxiv.org/abs/2108.08810) [[Summary]](https://github.com/SeongSuKim95/WORK/blob/master/Paper_review/3.2%20Do%20Vision%20Transformers%20See%20Like%20Convolutional%20Neural%20Networks.md)
   3. *__How Do vision Transformers Work?(ICLR,2022)__*[[LINK]](https://arxiv.org/abs/2202.06709) [[CODE]](https://github.com/xxxnell/how-do-vits-work)
   4. On the Expressive Power of Self-Attention Matrices
   5. (LayerNorm) Improved Robustness of Vision Transformer via PreLayerNorm in Patch Embedding
   6. (LayerNorm) On Layer Normalization in the Transformer Architecture
   7. Going deeper with Image Transformer (ICCV 2021, 2021/03/31) [[LINK]](https://arxiv.org/abs/2103.17239)
   8. Rethinking Spatial Dimensions of Vision Transformers (ICCV 2021) [[LINK]](https://openaccess.thecvf.com/content/ICCV2021/html/Heo_Rethinking_Spatial_Dimensions_of_Vision_Transformers_ICCV_2021_paper.html)
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
2. About Transformer (80%)
3. Nature of Transformer (80%)
4. Positional Encoding of Transformer (100%)
5. Injecting bias to Transformer (70%)
6. Using Transformer in Re-ID (80%)
7. Implementation & Details (100%) - 04/21 - 06/11 Finished
# KEY IDEA 

여러 논문들을 읽으면서 얻은 KEY IDEA 들을 적는다. Groups의 index로 논문을 표시한다.

## 1. Attention + CNN 
## 2. About Transformer 
## 3. Nature of Transformer  
  - ### 3.2 Do vision Transformers see like convolutional neural networks? [[Summary Link]](https://github.com/SeongSuKim95/WORK/blob/master/Paper_review/3.2%20Do%20Vision%20Transformers%20See%20Like%20Convolutional%20Neural%20Networks.md)
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
  - ### 4.3 __On position embeddings in BERT(ICLR 2021, 20/09/29)__ [[IDEA]](https://github.com/SeongSuKim95/WORK/blob/master/%EC%B6%94%EA%B0%80%20%EC%84%A4%EB%AA%85%20%EC%9E%90%EB%A3%8C/4.3%20On%20position%20embeddings%20in%20BERT.md)
  - ### 4.4  __Rethinking positional encoding in language pre-training (ICLR 2021, 20/06/28)__ [[IDEA]](https://github.com/SeongSuKim95/WORK/blob/master/%EC%B6%94%EA%B0%80%20%EC%84%A4%EB%AA%85%20%EC%9E%90%EB%A3%8C/4.4%20Rethinking%20positional%20encoding%20in%20language%20pre-training.md)
  - ### 4.5 __Do we Really Need Explicit Position Encodings for Vision Transformers?(21/02/22)__[[IDEA]](https://github.com/SeongSuKim95/WORK/blob/master/%EC%B6%94%EA%B0%80%20%EC%84%A4%EB%AA%85%20%EC%9E%90%EB%A3%8C/4.5%20Do%20we%20Really%20Need%20Explicit%20Position%20Encodings%20for%20Vision%20Transformers.md)
  - ### 4.6 __Rethinking and Improving Relative Position Encoding for Vision Transformer__ (ICCV 2021, 21/07/29)[[IDEA]](https://github.com/SeongSuKim95/WORK/blob/master/%EC%B6%94%EA%B0%80%20%EC%84%A4%EB%AA%85%20%EC%9E%90%EB%A3%8C/4.6%20Rethinking%20and%20Improving%20Relative%20Position%20Encoding%20for%20Vision%20Transformer.md)
## 5. Injecting bias to Transformer 
  - 5.5 Swin Transformer는 convolution 의 filter 개념을 window 라는 개념으로 치환하여 적용한 Transformer이다. 방법이 아무리 복잡하더라도, 개념적으로 쉽고 납득이 잘가면서 좋은 결과를 얻을 수 있다는 것이 매우 매력적이다. Window라는 개념이 ReID에 쓰인다면 어떻게 쓰여야 할까..? 사람의 특성을 생각하여 Vertical한 방향으로 Window를 확장해 나간다면..?
  - Window의 위치 정볼르 encoding하기 위해 relative positional embedding 기법이 사용되었다. 코드가 굉장히 직관적이고 구현이 쉬워서 이를 내 구현에 참고하고자 한다.
## 6. Using Transformer in Re-ID 
# IMPORTANT FACTS
  - Transformer를 이용한 Classification를 다루는 모든 논문의 benchmarking은 2.4(DeiT)와의 비교를 통해 이루어진다. ReID는 기본적으로 classification 이므로 참고해야할듯 하다.
  - Classification에서의 효율적인 positional encoding 방식은 relative positonal encoding이다. 이 과정에서 CLS token에 대한 고려는 반드시 이루어져야 한다. 
  - 조사해본 결과, 2D relative positional encoding의 효용성에 대해선 아직 밝혀지지 않은 것이 많다. 21/07/29에 ICCV 2021에 submit 된 4.6에서도 이를 명시적으로 언급하고 있는것으로 보아, 내가 열심히만 한다면 논리적인 무언가를 만들어 낼 여지가 많아 보인다. __아직 밝혀지지 않은 것일 뿐, 새로운 방법은 무조건 존재 할것 같다.__ 
  - 아쉽게도, 현재 성능이 좋은 ReID module 들은 아직 CNN에 의존하고 있다. ReID task가 많은 inductive bias를 요구한다는 반증이기도 하다. 연구할 여지가 많다는 점에서 좋은점일지도 모른다.
  - *ViT는 CLS token을 query의 입장에서, Patch token을 Query의 입장에서 optimize하도록 설계되어 있다.* --> 간단하지만 굉장히 중요! 
  - Patch token의 importance rank 
      - ViT의 학습 과정 중, CLS token은 self-attention의 mechanism상 patch token들의 weighted summation 형태로 update된다.
      - CLS token과의 attention score가 높은 patch token일수록, model은 해당 patch token에 localize된 feature로 class를 판별한다. 이는 ViT의 attention을 visualize하는 방식에도 쓰이는 개념이다.
      - CLS token이 classifier의 FC layer에 곱해진 후, 각 class에 대한 확률 값으로 변환된다는 점에서 classifier의 weight parameter들은 각 class에 대한 대표 feature를 나타내는 벡터가 된다. 이는 Elemented weight triplet loss 에서도 등장한 개념이다.
      - 따라서, 특정 class가 갖고 있는 일반적인 특성이 잘 반영된 patch를 찾기 위해 classifier의 weight parameter와 patch token간 similarity를 구한 후 ranking을 매기면 중요도 높은 patch를 구할 수 있다.
    - 구현 과정에서 알아낸 것들
      - CLS token - patch token similarity rank와 Classifier weight - patch token similarity rank를 비교해보면 결과가 꽤 유사하다. 즉, sample내에서 중요한 patch가 해당 class의 일반적인 특징을 가진 patch일 확률이 높다.
  
# Sketch
Idea ,Facts를 기반으로 Idea를 구상한다. 
 
### Idea 1: *Vision Transformer의 Self attention mechanism이 metric learning에 더 유용하도록 loss를 설계할 수 있는가?* 
### Idea 2 :  *Patch단위의 정보를 유지하는 vision transformer의 특성과, 사람의 신체 구조를 학습할 수 있는 relative postional encoding을 통한 inductive bias으로 ReID의 성능을 올릴 수 있는가?*
- 먼저, 3.2에 근거하여 output patch를 사용하는 것이 나아보인다. 어찌되었건 각 part image에 coressponding한 정보를 담고 있기 때문이다. 골라낼 수만 있다면, element weighted triplet hard loss 처럼 loss에 적용할 수도 있겠다.
  - 6.2 논문이 이 Idea를 기반으로 ReID를 수행하였으나, 방법만 제시하고 해석은 제시되어 있지 않다.
- Relative position을 사용해야한다. 이 과정에서 relative positional bias를 어떻게 사용하여 inductive bias를 modeling 하느냐가 관건이 되겠다.
  - Relative posion의 구현은 여러 형태가 존재하지만 Swin Transformer의 code(bias term)이 내 idea의 목적과 가장 어울리는 구현으로 보인다. 
- ViT의 기본 속성을 그대로 안고 갈 것이기 때문에, 추가적인 module이 필요하다면 ViT의 pretrained weight들의 optimize 방향에 반하지 않는 역할을 할 수 있는 위치에 plug-in 되어야 한다.

### 논리 전개 구상
  1. Vision transformer는 vision task에서 좋은 성능을 보여왔다.
  2. ReID task에 대해선 아직 CNN 기반의 모델이 dominant하다. 또, 아직 ViT의 nature를 ReID에 어떻게 활용할 수 있는지에 대한 연구는 많지 않다.
  3. ViT엔 CNN엔 없는 positional embedding이란 장치가 있다. 이는 patch의 위치 정보를 모델이 학습할 수 있도록 돕는다.
  4. 이 positional embedding은 크게 두 종류로 나뉘는데, APE와 RPE이다.
  5. RPE는 vision 영역에서 efficacy가 충분히 검증되지 않았다. 또, 이것의 학습이 image에 대해 충분히 이루어지는지도 미지수이다.
  6. 이는 image와 sentence의 구조적 차이가 충분히 반영되지 않았기 때문이다. 이를 해결하기 위해 RPE에 여러 기법들이 적용된 논문들이 있다. Swin transformer는 RPE를 bias의 형태로 단순화하여 적용하였을 때 성능적으로 우수함을 입증하였다.
  - ImageNet : ![Image_Net](https://user-images.githubusercontent.com/62092317/173336210-a5342cad-1156-4dea-9515-f6d40d1a3878.PNG)
  - ReID dataset : ![ReID_data](https://user-images.githubusercontent.com/62092317/173336651-456a6d96-d6af-4e3f-abca-46cf49cd80ca.PNG)
  7. Re-ID dataset은 여타 image dataset과 다르게, 모든 sample이 사람을 대상으로 하기 떄문에 sample간 correlation(즉, 형태적 유사성)이 높다. 즉 sample내 object의 형태가 비슷하기 때문에 patch의 위치가 갖는 의미가 다른 dataset에 비해 크다. 
  9. 따라서, PE가 갖는 특성을 활용해 ReID dataset을 효율적으로 학습할 수 있도록 explicit한 supervision을 주고자 한다.
  10. RPE가 갖는 특성을 활용하여 동일한 사람의 신체구조를 modeling한 loss를 설계 하고, 이것이 triplet 학습과정에서 반영되도록 한다. 
  11. 여러 ReID dataset(Market-1501, DukeMTMC-ReID, CHUK03-np, MSMT17-V2)에 대해 제안한 방식을 통한 성능향상이 있음을 확인했다.
# Working process with Time log 

- ## Code Repository [[LINK]](https://github.com/SeongSuKim95/TransReID)
- ## Structure of ViT based ReID   
![ViT_ReID](https://user-images.githubusercontent.com/62092317/173266708-e1180249-5e2f-4cda-aacd-f459c1ea980b.PNG)
  
- ## Visualization tools 
   - 0305 : Query image에 대한 Top 10 Rank gallery visualization 완료
   - 0306 : Query image에 대한 Attention roll out 완료 
   - 0310 : Query image에 대한 Visualize 결과 통합
     ![show](https://user-images.githubusercontent.com/62092317/157599820-cb30c46e-e4b0-4a95-9584-fa64866b0327.png)
   - 0315 :  
     - Training / Weight & Bias 를 통해 attention map 연동 완료
     - HARDEST QUERY 출력 완료
     - Positional Embedding visualize 완료
   - 0320 : Weight and Bias logging added, Elemented weighted triplet loss added
- ## Method 1
   - 0322 :
     - Patch wise Triplet loss 구현
       - Cosine distance FIX 완료
       - Euclidean distance 구현 완료
     - Self - Attention 과 metric learning의 연결고리..?()
       - 마지막 transformer layer에 anchor, negative, positive 간 self-attention이 고려된 부분이 추가된다면?
   - (0324-0325 commit hash 8375c8e,918eb32) Cross Attention 구조를 구현
   - 0329 : Weighted triplet branch
   - 0330-0401 : ID, Triplet loss experiment
      - Weighted Triplet loss가 ViT에서 효율적이지 않은 이유가 무엇인가? 
        - 일단 ID(CE loss)와 Triplet loss는 CLS token에 어떤 영향을 주면서 학습을 supervise하는지부터 확인
   - 0402 : Patch similarity를 기반으로 한 patch selecting algorithm 구현
   - 0406 : 학습이 진행됨에 따라(즉, Epoch이 늘어남에 따라), 학습의 근거가 되는 patch들의 비율을 점점 줄여나가도록 supervise
      - Ex) 초반 학습에선 sample의 50% patch만을 근거로 삼도록, 후반 학습에선 sample의 10% patch만으로도 학습이 이루어지도록
        ![Patch_Similarity](https://user-images.githubusercontent.com/62092317/173273611-fb8d167d-333a-4e91-a1e7-ce76dbca6fa0.PNG)
        - 기대 효과 - 모델이 similarity가 높은 patch에 더 집중할 수 있어, feature의 localization에 도움이 될것을 예상
        - 실험 결과 - mAP, Rank1 score 기준으로 학습에 큰 영향을 주지 않음
        - 원인 분석 - Simiarity 값이 softmax function을 통과하기 때문에, 모델은 이미 최상위 몇개의 patch에 집중한 상태이므로 하위 rank patch들의 영향력이 적음
   - *0411-0420 : Patch simliarity based Weighted Triplet loss 논리 최종 정리*
      - Triplet loss 는 Anchor sample과 Positive sample의 feature distance를 줄이도록 설계되어 있으나, 두 sample(A,P)의 공통된 feature가 dominant하기 때문에 학습 과정이 차이점에 집중하지 못한다.
      - 목표 : Anchor sample과 positive sample간 비교시 non-dominant feature를 가깝게 하는데 집중할 수 있도록 triplet loss를 설계
      - CLS token은 self-attention mechanism속에서 patch token feature의 weighted summation 의 형태로 학습된다는 점을 이용
        ![AP_similarity](https://user-images.githubusercontent.com/62092317/173285377-edd7935e-3260-4659-8ddd-c85181090223.PNG)
        ![Patch_similarity](https://user-images.githubusercontent.com/62092317/173285387-e1741bf0-3d6c-402c-9d77-2d929f94d1a7.PNG)
        1. Anchor CLS token - Anchor Patch tokens 간의 similarity를 계산, similarity가 높은 상위 50%의 anchor patch를 선별
        2. Positive CLS token - Anchor Patch tokens 간의 similarity를 계산, similarity가 높은 상위 50%의 anchor patch를 선별
        3. 1과 2에서 선별된 두 patch 집합의 교집합에 해당하는 patch 선별
        4. Anchor와 Positive의 CLS token에서 선별된 patch들의 weighted summation을 뺀 값(Non-dominant feature) 사용
      - 얻어낸 non-dominant feature를 normalize한 후, anchor CLS token의 각 feature에 element-wise로 곱합 weight vector를 생성
        ![Weighted_Triplet](https://user-images.githubusercontent.com/62092317/173286748-2cfe29dd-c3a7-4fb0-96fc-ea4cc6551a34.PNG)
      - Weight를 element-wise로 곱하는 것의 의미
        - Weight vector는 CLS token과 patch token간 similarity 기반으로 구함
        - Model이 triplet loss의 학습과정에서 Anchor sample과 Positive sample이 공통적으로 집중하는 patch들의 dominant feature를 제외한 나머지 부분을 효과적으로 배울수 있도록 지도
   - 0423 : TransReID-SSL[6.4]
      - 6.4의 baseline code와 code merge 완료
   - 0428 - 0430 : Non-dominant feature의 normalization method changed
  
  - ### (0513 추가) Method 1의 실패 원인 분석 
    - 제안한 방법을 적용하였을때, 여러 dataset에 대한 일관성이 확보 되지 않음을 알게 되었다. (일부 hyperparameter setting의 Market 1501 data에 대해서만 성능이 향상, 거의 다 하락)
    - *ViT의 기본적인 동작을 간과한 나의 실수* 이다.
    - 분석
      ![ViT_SA](https://user-images.githubusercontent.com/62092317/173295340-ab694654-281e-4a71-97b2-0713d5b46b84.PNG)
      - 그림 예시 : 4개의 patch token 에 대해 self-attention mechanism을 통해 output token이 생성되는 과정
      - *ViT의 모든 parameter는 CLS token을 Query의 입장으로, Patch token을 Key의 입장으로 optimize 한다.*
      - 각 token은 각자를 Query의 입장으로 SA를 수행하여 다음 layer로의 output을 생성하지만, ViT는 CLS token만을 가지고 loss를 구성하여 model을 supervise하기 때문에 *Query CLS token, Key patch token*에 중심을 두고 학습이 진행된다.
      - Method 1에서 Non-dominant feature를 뽑기 위해 CLS token과 patch token간 차를 구하게 된다. 이 후, 이를 기준으로 CLS token에 element-wise로 곱해질 weight vector를 얻는다. 이것이 CLS token에 곱해진 후, triplet loss를 구하게 되면 back propagation 과정에 patch token feature의 영향을 받게 된다.
      - 즉, patch token feature가 loss의 연산에 개입하게 되면서 Key 입장에서만 고려되었던 patch token들이 Query 입장에서도 고려되게 되어 Query CLS token 중심의 학습이 깨지게 된다. (ViT의 본래 학습 과정과 반대 방향으로 학습이 이뤄진다)
    - 결론 : *CLS token(Query) 중심의 ViT 학습에 반하지 않으려면 Patch token feature가 직접적인 supervision을 받지 않도록 loss를 설계 해야한다.*
    - (0601) Method 1 Drop 
- ## Method 2
  - Structure
    ![Structure](https://user-images.githubusercontent.com/62092317/174054133-fe75af45-3e2c-45e0-b56f-7adbeb81806a.PNG)
  - Motivation
    - ViT의 구조적 특성을 Re-ID의 학습과정에 적극적으로 반영해보자!
    - Patch 단위로 image를 처리하는 ViT를 이용하여 Re-ID dataset의 특성을 modeling 할 수 있지 있을까?
  - Idea
    - ViT의 unique한 구조 중 하나인 positional embedding을 사용하여, Re-ID sample에 대해 사람의 각 신체 부위를 담고 있는 patch간 상대적 위치 관계를 model이 학습할 수 있도록 하자.
    - 같은 ID의 sample은 신체 부위의 상대적 위치 관계가 비슷하다는 사실을 model에게 supervise, 즉 주요 patch들간 relative positional bias 분포가 비슷하도록 유도
  - 0503 : Relative positional embedding Added [[Link : About RPE]](https://github.com/SeongSuKim95/WORK/blob/master/%EC%B6%94%EA%B0%80%20%EC%84%A4%EB%AA%85%20%EC%9E%90%EB%A3%8C/Explanation%20of%20RPE.md)
  - 0505 : Position information modeling 설계 구현
    - 각 sample에 존재하는 important patch(Model이 집중하고 있는 patch)간 position 정보를 modeling하는 방식은 다음과 같다.
    ![Selecting_patch](https://user-images.githubusercontent.com/62092317/173366596-d88b797c-8df4-4b16-90b1-4cd0706c7f45.PNG)
    1. 각 Sample에 대해 model이 집중하는 patch token을 선별하기 위해 classifier weight와 patch token간 similarity를 구한다.[[Link : Classifier weight와 similarity를 구하는 이유]](https://github.com/SeongSuKim95/WORK/blob/master/%EC%B6%94%EA%B0%80%20%EC%84%A4%EB%AA%85%20%EC%9E%90%EB%A3%8C/Insight%20in%20Classifier%20layer.md)
    2. Similarity를 기반으로 중요도가 높은 patch들을 선별한다.
    3. 선별된 patch들간의 위치관계 정보를 absolute positional embedding과 Relative positional embedding으로 얻는다. Transformer의 self-attention mechanism을 고려하여, 두 patch의 APE 내적에 RPE를 더한 값을 사용하기로 했다.
    ![patch_information](https://user-images.githubusercontent.com/62092317/174057266-ff317234-43f2-488d-b040-c652ac34ea3d.PNG)
    ![positive_vector](https://user-images.githubusercontent.com/62092317/174062511-6ecdf60a-8886-4355-aa49-29bafe6d9b4d.PNG) 
    각 관계에 대해 생성한 값을 concat하여 하나의 tensor로 만든다.
  - 0506 : Wandb Sweep file created
  - 0507-0508 : 어떤 loss를 사용하여 supervise?
    - ![Pos_triplet](https://user-images.githubusercontent.com/62092317/174062089-57c14519-57f8-4b69-97f9-b274077fb6de.png)
    - 가장 먼저 든 생각은 triplet loss 이다. Anchor와 positive sample들에 대해 생성한 position 정보들을 head, layer 별로 concat하여 vector를 만들고 Euclidean distance를 최소화 하는 방향을 사용했다.
    - 실험결과, mAP 측정시 초반 학습이 매우 강했으나 학습 중반부터 학습이 오히려 떨어지더니 loss가 발산해버렸다.
    - ## Triplet loss의 실패 원인 분석
      - *Triplet loss의 성질과 position vector의 numerical한 특성을 고려하지 않았기 떄문*이다.
      - Triplet loss는 기본적으로 sample의 feature vector에 대해 적용되며, 이 feature vector를 euclidean space에 projection한 후 distance를 계산한다. 
      - 내가 추출한 position 정보는 Attention score에 기반한 값이기 떄문에, *feature가 아닌 similarity* 이다. Similarity는 Euclidean space에서 다루지 않고 probability density의 형태로 사용한다.
      - 따라서, Self-attention mechanism의 전개와 동일하게 softmax를 도입하여 확률 분포의 형태로 만들고 분포의 유사도를 학습 loss를 사용하는 것이 맞다.[[Link : Problem in using Triplet loss for similarity score]](https://github.com/SeongSuKim95/WORK/blob/master/%EC%B6%94%EA%B0%80%20%EC%84%A4%EB%AA%85%20%EC%9E%90%EB%A3%8C/Problem%20in%20using%20Triplet%20loss%20for%20similarity%20score.md)
      - Solution : KL divergence loss를 사용하자!
  - 0510 : Jensen-Shannon Divergence loss added
    - Anchor와 positive의 position vector 각각에 softmax를 취한 후 KL divergence를 구하자 loss가 발산하는 문제가 사라졌다.
    - KL divegence loss는 수식상 non-symmetric한 성질을 갖는다. 즉, Anchor와 positive가 switch 되었을 때 loss값이 다르다.
    - 나의 idea상에선 같은 ID의 sample에 대해서 동일한 supervision이 이루어지길 바라기 때문에 Symmetry가 보장된 Jensen-Shannon Divergence loss를 사용하였다.
  - 0515 : Head-wise JSD loss implemented
  - 0516 : DukeMTMC, MSMT17, OCC-Duke dataset class added
  - 0522 : Combination with replacement added
  - 0524 : y-axis wise Relative position bias added 
  - 0530 : cuhk03-np dataset class added
  - 0604 : CLS attention score untie
  - 0610 : Code clean up, Freeze
# Etc
  1. Matplotlib 사용법
   - Subplot 기본 [[LINK]](https://soooprmx.com/matplotlib%EC%9D%98-%EA%B8%B0%EB%B3%B8-%EC%82%AC%EC%9A%A9%EB%B2%95-%EB%B0%8F-%EB%8B%A4%EB%A5%B8-%EC%8B%9C%EA%B0%81%ED%99%94-%EB%9D%BC%EC%9D%B4%EB%B8%8C%EB%9F%AC%EB%A6%AC/)
   - Subplot 간격 설정 [[LINK]](https://steadiness-193.tistory.com/174)
   - cv2 Attention map in PLT [[LINK]](http://www.learningaboutelectronics.com/Articles/How-to-display-an-OpenCV-image-in-Python-with-matplotlib.php)

  2. Dealing with Pytorch Model params [[LINK]](https://comlini8-8.tistory.com/50) [[LINK]](https://tutorials.pytorch.kr/beginner/saving_loading_models.html)

  3. Torch.Detach [[LINK]](https://redstarhong.tistory.com/64)
  
  4. TSNE for Debug [[LINK]](https://learnopencv.com/t-sne-for-feature-visualization/)

  5. Re-ID dataset 사용시 주의점
    - Re-ID dataset은 여타 dataset과 다르게, 초상권 문제가 자주 발생하여 dataset의 version을 잘 확인해야한다. Version에 따라 사용이 금지된 경우도 있고, 연구 목적으로만 release한 version이 존재하는 경우도 있다. 
      - DukeMTMC(Duke Multi-Tracking Multi-Camera)
        - Duke 대학의 campus에서 학생들을 대상으로 찍은 dataset인데 초상권 문제로 2019.6.2 이후로 프로젝트가 중지되었다. 그러나 Re-ID 연구목적으론 DukeMTMC-ReID라는 이름으로 사용이 가능하다.
      - MSMT17-V2
        - TransReID-SSL[6.4] 논문에 나와있는 성능을 official code를 통해 재현하려고 하였는데, Market-1501은 재현이 잘되는 반면 MSMT17에 대해선 꽤 큰 폭으로 하락한 성능이 나오는 것을 확인하였다.
        - 처음엔 코드 설정의 문제인가 하고 hyperparameter및 configuration을 전부 확인해봤지만, 이상한점을 찾을 수 없었다.
        - Data loader 부분의 root directory 부분을 살펴보니, 내가 사용하고 있는 dataset과 경로가 다르게 설정되어있는 것을 알 수 있었다. 이는 [6.4]의 논문이 MSMT17을 사용한 반면, 나는 MSMT17-V2를 사용하여 발생한 차이였다. 
        ![MSMT17_V2](https://user-images.githubusercontent.com/62092317/173303286-687583e8-7f84-4819-8653-6ed8c191ef1a.PNG)
        - MSMT17-V2는 MSMT17과 모든 spec이 같지만, 위 그림 처럼 얼굴부분이 전부 모자이크 처리 되어있다. 모자이크 처리 되어있는 상태의 dataset을 가지고 실험을 진행했기 때문에 재현이 되지 않은 것은 당연하다.
      - Cuhk03-np
        - Cuhk03 또한 new protocol로 촬영된 version인 Cuhk03-np를 사용해야한다.
      - Benchmark : Paperswithcode에서는 각 dataset의 version을 고려하지 않고 성능 순위를 매긴다. 따라서, 각 논문이 어떤 version의 dataset을 써서 성능을 측정하였는지 확인해가며 benchmark를 하는 것이 중요하다.

