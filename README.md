# Stock Market Price Predictor with LSTM (keras)

  I developed this project using Keras to build a Recurrent Neural Network (RNN), specifically using Long Short-Term Memory (LSTM) units, aimed at predicting market prices for the upcoming day and foreseeing possible trends. The training data is directly sourced from [Market Watch](https://www.marketwatch.com/investing/index/spx/download-data), providing data points such as Open price, Close price, volume, etc. Initially we extract the data from the csv (Open price) and make the data in Numpy arrays and process the data in order to run it through the LSTM model.

  Once data is processed, we set up the LSTM model by deciding its structure, defining layers, and configuring parameters such as the number of units, dropout rates to prevent overfitting, and the return sequence option to manage the flow of sequential data. This setup ensures the model's adaptability and effectiveness in capturing temporal patterns. After receiving the predicted prices we can graph the results with the actual movement of the market allowing one to see the visual differences and further dial in the parameters if required.  The model can be enhanced by incorporating additional parameters like volatility and trading volume.



## Modules Used 

+ **Numpy** : used to create arrays/matrices to apply ML models
+ **Matplotlib.pyplot** : Helps graphing 2D graphs in IDLE itslef
+ **Pandas** : used for data wrangling and data manipulation 
+ **MinMaxScaler** (sklearn.preprocessing) : Processing data to prepare it for the model 
+ **tensorflow.keras.models** : Provides utilities for creating, managing, and training models 
+ **tensorflow.keras.layers** : Facilitates defining the architecture of deep learning models by stacking layers with specified 
 configurations




## Why RNN and LSTM for the specific problem 

RNNs and LSTMs are used for predicting stock prices because they are well-suited for handling sequential data and capturing temporal dependencies, which are critical in financial time series. 

+ **Sequential Data Handling**: RNNs process data in sequence building up on previous data points, making them ideal for stock prices that 
 depend on historical trends. 
+ **Long-Term Dependencies:** LSTMs address the vanishing gradient problem in standard RNNs, allowing them to learn and retain long-term 
 patterns, crucial for capturing extended market trends.
+ **Dynamic Patterns:** LSTMs are robust in modeling nonlinear relationships and market volatility, which often exhibit dynamic and complex behavior.



## Considerations While choosing the model to work with

Alternative RNN models that could be used for predicting stock prices include GRU (Gated Recurrent Units), Vanilla RNN, and Bidirectional RNN/LSTM. However, each has its disadvantages when applied to stock market prediction:

1. **GRU** : May not capture very complex patterns as effectively as LSTMs due to their simpler structure.
2. **Vanilla RNN** : Suffers from the vanishing gradient problem, leading to poor performance when modeling long-term dependencies in time series like stock prices.
3. **Bidirectional RNN/LSTM**: Not practical for real-time stock prediction since future data is unavailable; also computationally more expensive. 
4. **Simple Feedforward Neural Network with Lagged Features** : Does not inherently model time dependencies, leading to suboptimal performance in sequential tasks like stock prediction.

## Architecture of the Model 

![LSTM-architechture](https://github.com/user-attachments/assets/ad2ec3bc-a89c-4939-bacd-7bc2a228c4f5)

 The model function similar to image shown but it contains 50 parallel nuerons, and continues the process for 5 set of hidden layrers, 
 droping 20% of the neurons to prevent overfitting of the model. The ouput at all 't' is then ploted to comapre with the actual movement.
 The **tanh** activation function is chosen because it outputs values between -1 and 1, enabling the model to handle both negative and 
 positive inputs effectively. The **mean_squared_error** loss function is used as it penalizes larger errors more heavily, making it ideal for regression tasks like 
 market price prediction.


  
## Dialing Down the Parameters

Major parameters that can be dailed in for the specific LSTM: 

+ **Units (50)**:
 The number of LSTM neurons determines the dimensionality of the output and affects the model's capacity to learn patterns. Too few neurons 
 can lead to underfitting, while too many can increase the risk of overfitting and computational load.

+ **Dropout (0.2)**:
 Dropout randomly deactivates 20% of the neurons during training to prevent overfitting. Higher values promote generalization but might 
 limit learning, while lower values reduce regularization, risking overfitting.

+ **Epochs (100)**:
 The number of epochs determines how many times the model learns from the entire dataset. More epochs may improve learning but can lead to 
 overfitting if the model starts learning noise. Fewer epochs might result in underfitting.

+ **Batch Size (32**):
 Batch size affects the frequency of weight updates. Larger batches improve gradient stability but might miss subtle patterns, while smaller 
 batches capture finer details but can lead to noisy updates and slower convergence.



## Outputs of Running the Model

![ Output of the model for the presented values ](https://github.com/user-attachments/assets/8199d2e1-8c40-433a-9a5a-a9d802c5c0e0)


The model successfully identifies the overall trends in stock prices. It also aligns reasonably well with the peaks and valleys, indicating that the LSTM is capturing some temporal dependencies in the data.
However, the predictions lag slightly behind the actual values at certain points, which is typical of LSTM models due to their sequential nature and reliance on past information. Also that the minor details aren’t accounted for by the model.
