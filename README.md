# moments-vae

## TODO list

* (1) ~~CHECK and Generalize the computation of M and the evaluation of Q (Go to exp4 below)~~
    * (1.1) ~~Make toy example to check it works~~
    <img width="625" alt="Gaussian M2" src="https://user-images.githubusercontent.com/29488113/67434449-4318d800-f5b8-11e9-8437-0e398e78fbb8.png">
    <img width="667" alt="Gaussian M4" src="https://user-images.githubusercontent.com/29488113/67434479-4f049a00-f5b8-11e9-9b3b-c0b0e76ad01f.png">
    <img width="667" alt="mixture M2" src="https://user-images.githubusercontent.com/29488113/67434532-6a6fa500-f5b8-11e9-8793-46e1a92a5ace.png">
    <img width="707" alt="mixture M4" src="https://user-images.githubusercontent.com/29488113/67434559-73607680-f5b8-11e9-99e0-b24a9185594f.png">

    
* (2) Reorganize code in visualizator to make just visualizations, not computations (@marinalpo)
* (3) When (1) is finished, check results of Q for a matrix with more moments than just 4

## Results

### Exp3

The purpose of this exp3 was to see the reconstruction error and the difference between the likelihood for outliers and inliers.

Reconstruction error: 
<img width="2268" alt="Screen Shot 2019-10-18 at 1 33 31 PM" src="https://user-images.githubusercontent.com/29488113/67124644-f2bd0700-f1c0-11e9-95a2-c76b800a4c36.png">

Likelihood error:
<img width="2208" alt="Screen Shot 2019-10-18 at 1 43 40 PM" src="https://user-images.githubusercontent.com/29488113/67124682-0d8f7b80-f1c1-11e9-977f-54a93a252c85.png">

***We can see that the likelihood for inliers sometimes is the same than for outliers. That could be enhanced by extracting better features.***

The different distribution for different z_i *INLIER CASE*
<img width="2113" alt="Screen Shot 2019-10-18 at 4 06 26 PM" src="https://user-images.githubusercontent.com/29488113/67124905-90183b00-f1c1-11e9-9939-37d56933fd91.png">

<img width="2113" alt="Screen Shot 2019-10-18 at 4 11 19 PM" src="https://user-images.githubusercontent.com/29488113/67125084-fef59400-f1c1-11e9-9760-ddd527092d98.png">

<img width="2113" alt="Screen Shot 2019-10-18 at 4 12 46 PM" src="https://user-images.githubusercontent.com/29488113/67125153-25b3ca80-f1c2-11e9-9c47-3149d72550a0.png">

<img width="2113" alt="Screen Shot 2019-10-18 at 4 13 15 PM" src="https://user-images.githubusercontent.com/29488113/67125190-38c69a80-f1c2-11e9-8143-b572d907a13b.png">

<img width="2113" alt="Screen Shot 2019-10-18 at 4 13 53 PM" src="https://user-images.githubusercontent.com/29488113/67125223-4ed45b00-f1c2-11e9-87b9-fbdd93a786c5.png">

<img width="1757" alt="Screen Shot 2019-10-18 at 4 14 37 PM" src="https://user-images.githubusercontent.com/29488113/67125269-66abdf00-f1c2-11e9-9136-7ef2ac88b6d0.png">

The different distribution for different z_i *OUTLIER CASE*

<img width="2086" alt="Screen Shot 2019-10-18 at 4 16 01 PM" src="https://user-images.githubusercontent.com/29488113/67125341-99ee6e00-f1c2-11e9-97eb-6cc0aedc6cd4.png">

<img width="2086" alt="Screen Shot 2019-10-18 at 4 16 39 PM" src="https://user-images.githubusercontent.com/29488113/67125395-aecb0180-f1c2-11e9-9c1a-323a77d45f85.png">

<img width="2086" alt="Screen Shot 2019-10-18 at 4 17 25 PM" src="https://user-images.githubusercontent.com/29488113/67125566-1c772d80-f1c3-11e9-9c80-cf890a515cc5.png">

<img width="1755" alt="Screen Shot 2019-10-18 at 4 20 33 PM" src="https://user-images.githubusercontent.com/29488113/67125613-3b75bf80-f1c3-11e9-86f8-67d4df36920f.png">

### Exp4

The purpose of exp4 was to approximate the distribution of different zi with the matrix of moments, then find outliers using Q (cristoffel polynomial). ***IT DOES NOT WORK AND OCTAVIA SAID THAT MAYBE THERE IS A NUMERICAL PROBLEM WITH THE INVERSE OF THE MOMENT MATRIX***

<img width="640" alt="Screen Shot 2019-10-18 at 4 39 10 PM" src="https://user-images.githubusercontent.com/29488113/67126799-f30bd100-f1c5-11e9-86dd-5eca8e6acdbb.png">
