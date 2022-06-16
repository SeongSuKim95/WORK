# CLS token으로 학습되는 Classifer layer(FC layer)는 어떤 속성을 갖고 있는가?

- Classifier layer는 fully connected layer로, network의 출력 feature를 input으로 받아 train set의 class 개수에 해당하는 dimension의 크기를 가진 vector로 출력한다. ( Shape : (feat_dim, class_num))
- 이렇게 출력된 vector는 softmax function을 통과하여 각 class에 대한 probability값을 갖도록 바뀌게 되고, 이것과 ground truth label간 Cross-entropy를 최소화하는 방향으로 model은 supervision을 받는다.
![Classifier_weight](https://user-images.githubusercontent.com/62092317/173371665-3691861f-41b7-4dcb-a7db-33dd920cf313.PNG)
- 위 그림은 ResNet backbone으로 Market-1501 dataset을 학습할 떄, classifier weight와 feature vector간 연산을 보인 그림이다.
- 751은 Market-1501의 총 ID(class)개수 이며, 2048은 backbone의 feature dimension이다.
- Feature vector는 Classifier weight의 각 row에 대해 dot product연산을 수행하여 하나의 scalar 값을 출력하며, 이는 softmax function을 거쳐 probability로 class에 대한 변환 된다.
- Training이 거듭될수록, classifier의 weight의 각 row는 해당 class를 대표하는 feature vector와 같은 방향을 가지도록(feature에 대해 dot product 값이 커지도록) update 된다.
- 이러한 특성을 이용해서, [1]에선 classfier weight와 feature를 element wise로 비교한 근거로 non-dominant feature에 weight를 부여하여 triplet loss를 학습시켰다. 

[1] Y. Lv, Y. Gu, and L. Xinggao, "The Dilemma of TriHard Loss and an Element-Weighted TriHard Loss for Person Re-Identification," Advances in Neural Information Processing Systems, vol. 33, 2020.