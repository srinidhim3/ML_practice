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
        2. Thompson Sampling

    The Multi-Armed Bandit Problem : used to train robot dogs to walk using reinforcement learning. finding the best outcome out of many options available. 
        1. We have d arms. for example, arms are ads that we display to users each time they connect to a web page.
        2. each time a user connects to this web page, that makes a round.
        3. At each round n, we choose one ad to display to the user.
        4. At each round n, ad i gives reward. 1 if the user clicks on the ad i, 0 if the user didn't.
        5. our goal is to maximize the total reward we get over many rounds.

