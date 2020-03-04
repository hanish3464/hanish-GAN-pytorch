## Least Square Error = L2 Loss = MSE (Mean Square Error)

`실제 값과 예측 값들의 차의 제곱을 합한 것과 같다.`

![equation](https://user-images.githubusercontent.com/51018265/75888695-1da5b000-5e6f-11ea-9d47-41327ca77ac4.png)

## LSGAN의 Idea

`기존의 GAN 논문들의 BCE Loss를 사용하여 Discriminator를 잘 속이기만 하면 되었던 방식으로 weights를 update 하는 방식에서 실제 target의 분포에
가까워 지도록 거리의 오차를 줄이도록 하여 weights를 update는 방식이라 할 수 있겠다. 모든 코드에서 단 2줄이면 해결되는 간단한 idea 임에도 결과는 굉장히 훌륭. `

![lsgan_idea](https://user-images.githubusercontent.com/51018265/75888775-40d05f80-5e6f-11ea-9a0d-90fc98875059.png)
