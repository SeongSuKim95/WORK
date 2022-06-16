![Similarity_list](https://user-images.githubusercontent.com/62092317/174065645-0793e44c-2975-49eb-a30c-2050282fa7bc.PNG)
Similarity score를 쭉 늘여 놓은 두개의 list가 있다고 하자. 이 두개의 list를 비교하는 것은 두개의 feature vector를 비교하는 것과 개념적으로 어떤 차이가 있는가?
![Feature_example](https://user-images.githubusercontent.com/62092317/174067306-9c76f8d7-cf39-4887-af58-0ffef15b7aa2.PNG)

위 그림은 내가 생각한 굉장히 간단한 예시이다. 우리가 보통 model을 통해서 추출한 feature는 정확한 의미를 알 수 없더라도, 각각이 서로 orthogonal하다고 가정한다. 그렇기 떄문에 Euclidean space에서의 feature 비교 방식이 의미가 있는 것이다. 그러나, similarity 값들은 이런 numerical 한 특성을 가진 값들이 아님과 동시에 그것들을 모아놓은 list는 단순 값들의 나열일 뿐이다.

이들을 feature vector와 동일하게 여겨 Euclidean space로 가져가 distance를 구한다는 것은 nonsense이다. Numerical한 특성을 고려할 떄, similarity값들은 probability의 형태로 변환하여 사용하는 것이 타당하다.(Ex: softmax)