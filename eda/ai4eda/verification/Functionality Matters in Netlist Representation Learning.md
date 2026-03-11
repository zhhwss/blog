## Functionality Matters in Netlist Representation Learning

* node 是不同 gate
* edge 是不同的边
![](images/2023-12-11-17-51-42.png)
* 不同gate 用不同的MLP 来 aggerate
![](images/2023-12-11-17-53-07.png)
* loss 设计是对比学习
![](images/2023-12-11-17-54-07.png)
* 从input 到 output 逐渐传递
![](images/2023-12-11-17-54-44.png)