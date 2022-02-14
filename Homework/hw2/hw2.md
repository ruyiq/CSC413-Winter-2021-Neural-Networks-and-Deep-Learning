## Homework 2
### 1. Optimization
#### 1.1 Mini-Batch Stochastic Gradient Descent (SGD)
##### 1.1.1 Minimum Norm Solution
Recall from Question 3.3.2 from HW1, we find that the solution obtained by gradient descent is $w^* = (X^TX)^{-1}X^Tt$ for $Xw^*=t$.

Let $w_0=0, d>n$. 
Assume mini-batch SGD converges to a solution $\hat{w}$ such that $X\hat{w} =t$.

WTS: $\hat{w} = w^*$ 

Since $x_j$ is the jth row of martix $X$, we know that $x_j$ is contained in the span of $X$. 

$$\begin{align*}
\frac{1}{b}\triangledown_{w_t} \mathbf L(x_j,w_t) &=  \frac{1}{bn} \frac{\partial}{\partial w_t} ||x_jw_t-t_j||^2_2 \\
&= \frac{2}{bn} x_j^T(x_jw_t-t_j)
\end{align*} $$

Notice that the gradient is spanned by the rows of $X$. 

Now, we need to prove convergence of weights by setting the gradient of loss to weight to zero. 
$$\begin{align*}
\frac{2}{bn} x_j^T(x_jw_t-t_j) &= 0 \\
x_j^T(x_jw_t-t_j) &= 0 \\
x_j^T x_jw_t - x_j^Tt_j &= 0 \\
x_j^T x_jw_t &=  x_j^Tt_j \\
w_t &= \frac{x_j^Tt_j}{x_j^T x_j} \\
&= \frac{t_j}{x_j^T x_j}x_j^T 
\end{align*} $$

Notice that $t_j \in \mathbf{R}$ and $x_j^T x_j \in \mathbf{R} $. Therefore, $\frac{t_j}{x_j^T x_j} \in \mathbf{R}$. Let $c= \frac{t_j}{x_j^T x_j}$. Then, we have $w_t = cx_j^T $.  

Clearly, the update steps of mini-batch SGD never leavess the span of $X$. Thus, we can say that every updated weight can be wrriten in terms of a linear combination of rows of $X$. 

We can thus write $\hat{w} = \mathbf{X}^Ta$ for some $a \in \mathbf{R}^n$. Thus, $$X\hat{w}-t = \mathbf{X}\mathbf{X}^Ta-t = 0$$
Therefore, 
$$\begin{align*}  \mathbf{X}\mathbf{X}^Ta &= -t\\
a &= (\mathbf{X}\mathbf{X}^T)^{-1}t  && \text{Since  when $n>d$, $XX^T$ is invertible}  \\
\mathbf{X}^Ta &= \mathbf{X}^T(\mathbf{X}\mathbf{X}^T)^{-1}t\\
\hat{w} &= w^*
\end{align*}$$



#### 1.2 Adaptive Methods
#####  1.2.1 Minimum Norm Solution
Let $d>n$.
Assume the RMSProp optimizer converges to a solution. 
As hinted, let $x_1= [2,1]$, $w_0=[0,0]$, $t=[2]$.
As clarified in piazza @503, $x_1$ is a row in the data matrix $X$ and $w_0 $ is a column vector.

$$\begin{align*} 
w^* &= \mathbf{X}^T(\mathbf{X}\mathbf{X}^T)^{-1}t \\
&=  x_1(x_1^Tx_1)^{-1}t \\
&= x_1 \cdot \frac{1}{5} \cdot 2 \\
&= \frac{2}{5} x_1
\end{align*}$$

Thus, the minimum norm solution is $\frac{2}{5}x_1$.
For the RMSProp optimizer: 

$$\begin{align*} \triangledown_{w} L 
&= \frac{2}{n}x_1(x_1^Tw-t)\\
&= \frac{2}{n} (x_1x_1^Tw-x_1t) \\
&= \frac{2}{n} \left[\begin{pmatrix}2\\1\end{pmatrix}\begin{pmatrix}2&1\end{pmatrix}w - \begin{pmatrix}2\\ 1\end{pmatrix}t\right] \\
&= \frac{2}{n}\left[\begin{pmatrix}4&2\\ 2&1\end{pmatrix}w-\begin{pmatrix}2\\ 1\end{pmatrix}t\right] \\
&= \frac{2}{n}\left[\begin{pmatrix}4&2\\ 2&1\end{pmatrix}w-\begin{pmatrix}4\\ 2\end{pmatrix}\right] \\
&= \frac{2}{n}\left[\begin{pmatrix}4&2\\ 2&1\end{pmatrix}w- 2 x_1\right]
 \end{align*} $$

Let $n=1$. Then, we have $\triangledown_{w} L=\left[\begin{pmatrix}8&4\\ 4&2\end{pmatrix}w- 4 x_1\right]$

Then, we need to check whether it converges to the minimum norm solution. Inspired by piazza @460, I decided to write some code to see what the RMSProp converges to.
### 2. Gradient-based Hyper-parameter Optimization
#### 2.1 Computation Graph
##### 2.1.1 
##### 2.1.2 
#### 2.2 Optimal Learning Rates
##### 2.2.1
##### 2.2.3
#### 2.3 Weight decay and L2 regularization
##### 2.3.1
##### 2.3.2
### 3. Convolutional Neural Networks
#### 3.1 Convolutional Filters
#### 3.2 Size of Conv Nets
#### 3.3 Receptive Fields