k-means clustering
    step 1 : choose the number of K clusters
    step 2 : select at random K points, the centroids ( not necessarily from your dataset).
    step 3 : Assign each data point to the closest centroid --> that forms K clusters.
    step 4 : compute and place the new centroid of each other
    step 5 : reassign each data point to the new closest centroid. if any reassignment 
                took place then go to step 4 otherwise go to FIN.
    
    Random Initialization Trap : the selection of centroid at the beginning of algorithm dictates the output. To correct this we need to use K-Means++ algorithm. There is no need to implement k-means ++ because its taken care of by library.

    Choosing right number of clusters : WCSS (Within Cluster Sum of Squares) Square of Distance between taken point to centroid. selecting more clusters will result in less value of WCSS. the upper limit to select the number of clusters is number of data points which would lead us to see WCSS as zero. To get optimal number of clusters we need to use Elbow method. its the value change in WCSS while increasing the number of clusters.

Hierarchical clustering : two types are Agglomerative and Divisive. below steps are for Agglomerative.
    step 1 : make each data point a single point cluster that forms N clusters.
    step 2 : Take two closest data points and make them one cluster, that forms N-1 clusters.
    step 3 : take two closest clusters and make them one cluster, that forms N-2 clusters.
    step 4 : repeat step 3 until there is only one cluster.

    Euclidean distance is used for calculating the distance between data points/clusters. Distance between clusters can be calculated in multiple ways.
        1. Closest points
        2. Furthest points
        3. Average distance
        4. distance between centroid.
    
    Dendograms : Memory of hierarchical model. The distance between two data point/clusters is proportional to the height of dendograms plot. by setting the height level threshold we can decide how many clusters we need. if the threshold level is disected by two lines then the total number of clusters will be two.

Association Rule learning (Apriori): People who bought also bought. 
    Apriori Support : People who watched movie 1 also watched movie 2
    Apriori Confidence : percentage of people who watched movie 2 
    Apriori lift : percentage of possible suggestion of movie 2 to the people who watched movie 1 already.

    Step 1 : set a minimum support and confidence
    Step 2 : take all the subsets in transactions having higher support than minimum support.
    Step 3 : take all the rules of these subsets having higher confidence than minimun confidence.
    Step 4 : sort the rules by decreasing lift.

Association Rule learning (Eclat) : 
    Eclat Support : users watchlist containing M / users watchlist

    Step 1 : set a minimum support
    Step 2 : Take all the subsets in transactions having higher support than minimun support.
    Step 3 : sort these subsets by decreasing support.

Reinforcement Learning:
    Reinforcement Learning is a powerful branch of Machine Learning. It is used to solve interacting problems where the data observed up to time t is considered to decide which action to take at time t + 1. It is also used for Artificial Intelligence when training machines to perform tasks such as walking. Desired outcomes provide the AI with reward, undesired with punishment. Machines learn through trial and error.

    In this part, you will understand and learn how to implement the following Reinforcement Learning models: 
        1. Upper Confidence Bound (UCB)
            Initial round we will start with default confidence level. as the go through more rounds this level is adjusted based on reward.
            confidence bound is also adjusted based on reward. we start the next round with highest confidence bound. when the bound decrease to a minimum level then we will select that option.
        2. Thompson Sampling : probablistic models
            we are not trying to guess the distributions behind the machines/data set. 
            every round there will be a feedback using which something new learnt. meaning the distribution is adjusted.


    The Multi-Armed Bandit Problem : used to train robot dogs to walk using reinforcement learning. finding the best outcome out of many options available. 
        1. We have d arms. for example, arms are ads that we display to users each time they connect to a web page.
        2. each time a user connects to this web page, that makes a round.
        3. At each round n, we choose one ad to display to the user.
        4. At each round n, ad i gives reward. 1 if the user clicks on the ad i, 0 if the user didn't.
        5. our goal is to maximize the total reward we get over many rounds.

    UCB vs Thompson sampling :
        1. UCB is deterministic model. but thompson is more of probabilistic.
        2. UCB requires update at every round. Thompson can accommodate delayed feedback.
        3. Thompson has better empricial evidence.

Natural Language Processing
    Natural Language Processing (or NLP) is applying Machine Learning models to text and language. Teaching machines to understand what is said in spoken and written word is the focus of Natural Language Processing. Whenever you dictate something into your iPhone / Android device that is then converted to text, that’s an NLP algorithm in action.You can also use NLP on a text review to predict if the review is a good one or a bad one. You can use NLP on an article to predict some categories of the articles you are trying to segment. You can use NLP on a book to predict the genre of the book. And it can go further, you can use NLP to build a machine translator or a speech recognition system, and in that last example you use classification algorithms to classify language. Speaking of classification algorithms, most of NLP algorithms are classification models, and they include Logistic Regression, Naive Bayes, CART which is a model based on decision trees, Maximum Entropy again related to Decision Trees, Hidden Markov Models which are models based on Markov processes.A very well-known model in NLP is the Bag of Words model. It is a model used to preprocess the texts to classify before fitting the classification algorithms on the observations containing the texts.

    Exampels
        1. if / else rules for chatbots
        2. Audio frequency components and analysis for speech recognition
        3. Bag-of-words model (classification)
        4. CNN for text recognition (classification)
        
Deep Learning is the most exciting and powerful branch of Machine Learning. Deep Learning models can be used for a variety of complex tasks:

    Artificial Neural Networks for Regression and Classification
    Convolutional Neural Networks for Computer Vision
    Recurrent Neural Networks for Time Series Analysis
    Self Organizing Maps for Feature Extraction
    Deep Boltzmann Machines for Recommendation Systems
    Auto Encoders for Recommendation Systems

    Training the ANN with stochastic gradient descent:
        step 1: randomly initialize the weights to small numbersclose to 0 but not 0.
        step 2: input the first observation of your dataset in the input layer, each feature in one input node.
        step 3: forward propagation from left to right the neurons are activated in a way that the impact of each neuron's activation is limited by the weights. propagate the activations until getting the predicted result y.
        step 4 : compare the predicted result to the actual result. Measure the generated error.
        step 5 : back-propagation from right to left the error is back propagated. update the weights according to how much they are responsible for the error. the learning rate decides by how much we update the weights.
        step 6 : repeat steps 1 to 5 and update the weights after each observation (Reinforcement learning). or repeat steps 1 to 5 and update the weights only after a batch of observations (Batch learning)
        step 7 : when the whole training set passed through the ANN, that makes an epoch. Redo more epochs.

Convolutional Neural Networks : used in image recognition.
    Step 1 : Convolution - find feature in image using feature detector and produce feature map.
                ReLu Layer - apply rectifier function on convolution layer to reduce the linearity of the images. Because images have lot of non linearity data.
    Step 2 : Max pooling - Spacial invariance. taking the max/sum value after grouping pixels data. this will result in reducing the number of features going into network. This will help in reducing the over fitting. If mean value is considered then its called sub-sampling.
    Step 3 : Flattening - put data from matrix to one column (long vector) which inturn becomes one feature to feed into ANN.
    Step 4 : Full Connection - ANN

    Softmax and cross-entropy : its another way to calculate last function. if there are multiple output neuron then probability adding up to 1 between their output is less. hence we use softmax equation to normalize the probability so that the sum will be 1. if there are two NN and one of them is out performing another at every instance. Comparing these two NN there can be 
        1. classification error which is either true or false kind of output. These is not a great way to compare the two NN.
        2. Mean Squared Error is taking the sum of squared errors and then just take the average across observations. This is more accurate in comparing two NN.
        3. Cross-Entropy is a great way to compare beacuse its based on logarithm. Hence the tiny variation will be captured. This is where mean squared error will fail to capture the small adjustments.Mean squared is good for regression, and cross-entropy is good for classification.

Dimensionality Reduction : There are two types of Dimensionality Reduction techniques
    1. Feature Selection : Feature Selection techniques are Backward Elimination, Forward Selection, Bidirectional Elimination, Score Comparison and more. We covered these techniques in Part 2 - Regression.
    2. Feature Extraction : In this part we will cover the following Feature Extraction techniques:
        1. Principal Component Analysis (PCA)
        2. Linear Discriminant Analysis (LDA)
        3. Kernel PCA
        4. Quadratic Discriminant Analysis (QDA)

Principal component analysis (PCA): unsupervised learning and used in below areas.
    1. Noise filtering
    2. Visualization
    3. Feature Extraction
    4. Stock market analysis
    5. Gene data analysis

    Goal of PCA is to identify patterns in data and detect the correlation between variables. Reduce the dimensions of a d-dimensional dataset by projecting it onto a k-dimension subspace where k < d

    