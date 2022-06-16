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