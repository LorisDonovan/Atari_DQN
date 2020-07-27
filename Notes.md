# Algorithm
Initialize replay memory $D$ to capacity $N$\
Initialize action-value function $Q$ with random weights $\theta$\
Initialize target action-value function $\hat{Q}$ with random weights $\theta^- = \theta$\
**for** episode $= 1, \ M$\
&emsp; Initialize sequence $s_1  = \{x_1\}$ and preprocessed sequenced $\phi_1 = \phi(s_1)$\
&emsp; **for** $t = 1,\ T$ **do**\
&emsp; &emsp; With probability $\epsilon$ select a random action $a_t$\
&emsp; &emsp; otherwise select $a_t = argmax_a Q(\phi(s_t), a; \theta)$\
&emsp; &emsp; Execute action $a_t$ in emulator and observe reward $r_t$ and image $x_{t+1}$\
&emsp; &emsp; Set $s_{t+1} = s_t, a_t, x_{t+1}$ and preprocess $\phi_{t+1} = \phi(s_{t+1})$\
&emsp; &emsp; Store transition $(\phi_t, a_t, r_t, \phi_{t+1})$ in $D$\
&emsp; &emsp; Sample random minibatch of transitions $(\phi_t, a_t, r_t, \phi_{t+1})$ from $D$\
&emsp; &emsp; Set $y_j = r_j$ if episode terminates at step $j+1$ &emsp;:&emsp; $r_j + \gamma\ max_{a'} \hat{Q}(\phi_{j+1}, a'; \theta^-)$ otherwise\
&emsp; &emsp; Perform a gradient descent step on $(y_j − Q(\phi_j, a_j; \theta))^2$ with respect to network parameters $\theta$\
&emsp; &emsp; Every $C$ steps reset $\hat{Q} = Q$\
&emsp; **end for**\
**end for** 


# Preprocessing
* Raw frame = 210x160 RGB
* Raw frames converted to gray-scale and down-sampled to 110x84 image
* then cropped to 84x84 region of the image, capturing the playing area (2D convolutions expect square input)
* apply preprocessing to the last 4 frames of a history and stack them to produce input to the Q-function 

# Model Architecture
* the model produces separate output unit for each possible action, and only the state representation is an input to the network
* the output corresponds to the predicted Q-values of the individual action for the input state
* input consists is 84x84x4 image produced by $\phi$ 
* the first hidden layer convolves 16 8x8 filters with stride 4 and applies ReLU
* the second hidden layer convolves 32 4x4 filters with stride 2 and applies ReLU
* the final hidden layer is fully connected and consists of 256 ReLU units
* the output is fully connected layer with a single output for each valid action

|Layer|Architecture|Activation|
|:---:|:---:|:---:|
|input |$4 \ \ 84 \times 84$ images| - |
|$1^{st}$ hidden layer|conv2D: $16 \ \ 8 \times 8$ filters; stride = $4$|ReLU|
|$2^{nd}$ hidden layer|conv2D: $32 \ \ 4 \times 4$ filters; stride = $2$|ReLU|
|$3^{rd}$ hidden layer|Fully Connected: $256$|ReLU|
|output layer|Fully Connected: $num\_actions$|-|

# Experiments
* fix all positive rewards to be +1 and all negative rewards to be -1, leaving 0 rewards unchanged. This limits scale of error derivatives and makes it easier to use the same learning rate across multiple games
* RMSProp algorithm with minibatches of size 32
* behavior policy was $\epsilon$-greedy with $\epsilon$ anealed linearly from 1 to 0.1 over the first million frames, and fixed at 0.1 thereafter
* Trained for a total of 10 million frames
* Replay memory of one million most recent frames
* The agent sees and selects action on every $k^{th}$ frame, and its last action is repeated on skipped frames. Here $k=4$

# Training and Stability
## Evaluation metric
* total reward the agent collects in an episode or game averaged over a number of games
* another metric is the policy's estimated action-value function Q, collect a fixed set of states by running a random policy before training starts and track the average of the maximum predicted Q for these states
