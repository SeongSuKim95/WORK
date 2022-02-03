# Purpose of repository

- 다음과 같은 5단계로 논문을 읽어나가고 담겨있는 내용과 idea를 정리한다. 
  
0. Attention + CNN
1. Transformer and Variants
2. Nature of Transformer
3. Positional Encoding of Transformer
4. Injecting bias to Transformer via structure change
5. Using Transformer in Re-ID

- 단계별로 앞으로 읽을 논문과 읽은 논문은 정리한다. 논문이 나온 시점과 다른 논문들과의 관계를 정리한다.
- 각 논문의 실험결과와 주장들을 근거로 삼아 idea를 구성 한다.   

# Short-term plan
 일주일 단위로 구성하는 단기 계획.   __읽은 논문은 bold체로 표시.__
- 1/29 ~ 02/05 Positional Encoding
  - __On the relationship between self-attention and convolution layers (ICLR 2020)__ [LINK](https://arxiv.org/abs/1911.03584)
    - __Supplementary__ [LINK](http://jbcordonnier.com/posts/attention-cnn/)
  - __Can Vision Transformer Perform Convolution? (ICLR 2022 underreview, 2021/11/02)__ [LINK](https://arxiv.org/abs/2111.01353)
  - __On position embeddings in BERT(ICLR 2021, 20/09/29)__ [LINK](https://openreview.net/forum?id=onxoVA9FxMw)
  - __Rethinking positional encoding in language pre-training (ICLR 2021, 20/06/28)__ [LINK](https://arxiv.org/abs/2006.15595)
  - __Do we Really Need Explicit Position Encodings for Vision Transformers? (21/02/22)__ 
    - Conditional Positional Encodings for Vision Transformers (21/03/18 revised version) [LINK](https://arxiv.org/abs/2102.10882)
  - __Rethinking and Improving Relative Position Encoding for Vision Transformer(ICCV 2021, 21/07/29)__ [LINK](https://arxiv.org/abs/2107.14222)
  - Stand-Alone self-Attention in Vision models (NIPS 2019, 2019/06/13) [LINK](https://arxiv.org/abs/1906.05909)
  - __Self-Attention with Relative Position Representations (NAACL 2018, 2018/03/06)__ [LINK](https://arxiv.org/abs/1803.02155)
  - What do position embeddings Learn? An Empirical Study of Pre-Trained Language Model Positional Encoding (EMNLP 2020, 2020/09/28) [LINK](https://arxiv.org/abs/2010.04903)
  - Improve Transformer Models with Better Relative Position Embeddings(EMNLP 2020 ,20/09/28) [LINK](https://arxiv.org/abs/2009.13658)
- 02/05 ~ 02/11 Injecting bias to Transformer
  - Visformer : The Vision-friendly Transformer (ICCV 2021, 2021/12/18) [LINK](https://arxiv.org/abs/2104.12533)
  - ConViT: Improving Vision Transformer with Soft Convolutional Inductive Biases
  - CMT : Convolutional Neural Networks Meet Vision Transformers
  - LocalViT : Bringing Locality to Vision Transformers
  - __Swin Transformer : Hierarchical Vision Transformer using Shifted Window__
  - CvT : Introducing Convolutions to Vision Transformers

# Groups

## 개념별로 논문들을 분류한다. 
*읽은 것은 이탤릭체*  
__중요한 논문은 bold체__

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
   2. *__Do vision transformers see like convolutional neural networks?__*
   3. On the Expressive Power of Self-Attention Matrices
   4. (LayerNorm) Improved Robustness of Vision Transformer via PreLayerNorm in Patch Embedding
   5. (LayerNorm) On Layer Normalization in the Transformer Architecture
4. Positional Encoding of Transformer
   1. *__On the relationship between self-attention and convolution layers (ICLR 2020)__* [LINK](https://arxiv.org/abs/1911.03584)
      - Supplementary [LINK](http://jbcordonnier.com/posts/attention-cnn/)
   2. *Can Vision Transformer Perform Convolution? (ICLR 2022 underreview, 2021/11/02)* [LINK](https://arxiv.org/abs/2111.01353)
   3. *__On position embeddings in BERT(ICLR 2021, 20/09/29)__* [LINK]()
   4. *Rethinking positional encoding in language pre-training (ICLR 2021, 20/06/28)* [LINK](https://arxiv.org/abs/2006.15595)
   5. *Do we Really Need Explicit Position Encodings for Vision Transformers?* (21/02/22) 
    - Conditional Positional Encodings for Vision Transformers (21/03/18 revised version) [LINK](https://arxiv.org/abs/2102.10882)
   6. *__Rethinking and Improving Relative Position Encoding for Vision Transformer__*(ICCV 2021, 21/07/29) [LINK](https://arxiv.org/abs/2107.14222)
   7. *Stand-Alone self-Attention in Vision models* (NIPS 2019, 2019/06/13) [LINK](https://arxiv.org/abs/1906.05909)
   8. Self-Attention with Relative Position Representations (NAACL 2018, 2018/03/06) [LINK](https://arxiv.org/abs/1803.02155)
   9. What do position embeddings Learn? An Empirical Study of Pre-Trained Language Model Positional Encoding (EMNLP 2020, 2020/09/28) [LINK](https://arxiv.org/abs/2010.04903)
   10. Improve Transformer Models with Better Relative Position Embeddings(EMNLP 2020 ,20/09/28) [LINK](https://arxiv.org/abs/2009.13658)
5. Injecting bias to Transformer via structure change
   1. Visformer : The Vision-friendly Transformer (ICCV 2021, 2021/12/18) [LINK](https://arxiv.org/abs/2104.12533)
   2. ConViT: Improving Vision Transformer with Soft Convolutional Inductive Biases
   3. CMT : Convolutional Neural Networks Meet Vision Transformers
   4. LocalViT : Bringing Locality to Vision Transformers
   5. Swin Transformer : Hierarchical Vision Transformer using Shifted Window
   6. CvT : Introducing Convolutions to Vision Transformers
6. Using Transformer in Re-ID 

# Supplementary 
- Transformer
  - Illustrated Transformer [LINK](https://jalammar.github.io/illustrated-transformer/)
  - Illustrated ViT [LINK](https://medium.com/analytics-vidhya/illustrated-vision-transformers-165f4d0c3dd1)
  - Swin Transformer [LINK](https://www.youtube.com/watch?v=2lZvuU_IIMA)
- Positional Embedding
  - Master Positional Encoding : Part 1 [LINK](https://towardsdatascience.com/master-positional-encoding-part-i-63c05d90a0c3)
  - Master Positional Encoding : Part 2 [LINK](https://towardsdatascience.com/master-positional-encoding-part-ii-1cfc4d3e7375) 
  - AI coffee break playlist [LINK](https://www.youtube.com/watch?v=1biZfFLPRSY&list=PLpZBeKTZRGPOQtbCIES_0hAvwukcs-y-x)
  
# Process

## 0. Attention + CNN 
## 1. About Transformer (100%)
## 2. Nature of Transformer (60%)
## 3. Positional Encoding of Transformer (70%)
## 4. Injecting bias to Transformer (0%)
## 5. Using Transformer in Re-ID (0%)


# IDEA 

## Facts
여러 논문들을 읽으면서 근거를 충분히 얻은 사실들을 적는다.

## Sketch
Facts를 기반으로 Idea를 구상한다.
ddddd
## Implemetation
구상한 Idea를 구현해보고 결과를 확인한다.



