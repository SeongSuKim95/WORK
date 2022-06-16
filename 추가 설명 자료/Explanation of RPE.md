ViT 의 positional Embedding method는 크게 Absolute positional embedding과 Relative positional embedding 2가지로 나뉜다. 
![PE](https://user-images.githubusercontent.com/62092317/173345736-d8ff84b1-34e8-4d12-a522-5d48afe5412d.PNG)
# 1. Absolute positional embedding
- Absolute positional embeddding은 각 patch에 대해 image의 어느 위치에 있었는지를 model이 학습할 수 있도록 고안된 learnable parameter
- Transformer layer에 들어가기 전에 Patch token feature에 더해진다. 처음 patch의 위치 정보는 모델의 학습과정 속에서 일관되게 유지되어야 하므로, 맨 첫 layer에서 1번 더해진다.
- 각 patch 마다 서로 다른 positional embedding을 가지며, 더해지기 떄문에 patch token feature와 dimension이 같다. 즉, APE의 shape은 (patch token + 
1, patch dimension) 이다. (1을 더하는 이유는 CLS token에도 APE가 존재하기 떄문)
# 2. Relative positional embedding
- Relative positional embedding은 APE와 다르게, 두 patch간 상대적인 위치 관계를 modeling하기 위한 learnable parameter이다.
- 상대적인 위치 관계라는 개념이 생소하므로 예시를 통해 그림으로 설명하고자 한다
![RPE1](https://user-images.githubusercontent.com/62092317/173348788-d07e77d3-8178-4bd7-b458-7b13a0f6fd7c.PNG)
- 위 그림 처럼 Image를 4개의 patch A,B,C,D로 나눴다고 가정해보자. Image가 2D grid형태인 것을 고려하여, 2D APE를 설정하면 좌상단 그림 처럼 각 patch 에 대한 PE를 지정할 수 있다. 
- 각 Patch 쌍에 대해 자신의 PE에서 상대의 PE를 빼면, 상대적인 position 거리를 구할 수 있다.
![RPE2](https://user-images.githubusercontent.com/62092317/173348790-25cdc502-5aca-4971-92c7-94ec6d8d9954.PNG)
- 모든 patch 쌍의 상대적 거리를 4 x 4  matrix로 표현할 수 있는데, 이를 scalar값으로 표현하기 위해 양의 값으로 만든 후 더한다.
- 이럴 경우 문제가 생기게 되는데, x축과 y축간 구분이 사라져 서로 다른 축에서 같은 거리만큼 떨어진 관계가 같은 값으로 표현된다.(Ex: (1,0)->1 , (0,1)->1)
![RPE3](https://user-images.githubusercontent.com/62092317/173348777-390bb6f3-29b5-4688-b080-60a5b94c3c3f.PNG)
- 따라서, 하나의 축에 대한 값을 scaling 한 후에 두 값을 더해줌으로써 이를 구분한다. 그림 같은 경우 x축에 대한 값을 2M-1 (M은 #row)배 해준 후 더하였다.
- 이렇게 구해진 값은 각각이 서로 다른 Relative position 관계를 표현한다. 같은 값일 경우 두 patch간 상대적인 위치 관계가 같음을 의미 한다. 예를 들어, 그림에서 A-B와 C-D의 Relative position은 3으로 원래 patch의 위치를 생각하면 알 수 있듯이 "자신을 기준으로 한칸 오른쪽" 위치를 의미한다.
![RPE4](https://user-images.githubusercontent.com/62092317/173348784-5944d058-f7fd-4a7a-a65d-78d2f5bdb03c.PNG)
- 각 Relative position 에 대한 값은 하나의 learnable parameter로 지정하여 모델이 알맞은 값을 학습하게 된다. 이 과정속에서 RPE는 Look up table형태의 table에 저장한 채로 학습하게 되는데, 이는 같은 relative position 을 갖는 patch 쌍에 대해선 같은 값이 학습되어야 하기 때문이다. 그림에서 볼수 있듯이 , 총 9개의 relative position을 학습하기 위해 크기가 9인 parameter table을 만들고 해당하는 위치에 반영한다.
- Patch의 개수가 N x M 이라고 할때, 가능한 relative position은 (2N-1) X (2M-1)개이다.
- 이렇게 생성된 relative position matrix는 transformer SA 내부의 patch간 관계를 나타내는 attention matrix에 bias의 형태로 더해지게 된다. (Method in Swin-Transformer)
- RPE를 구현하는 방식은 다양하지만, 모든 방식이 self-attention mechanism 내부에 RPE를 반영한다는 점은 동일하다.