k-means clustering
    step 1 : choose the number of K clusters
    step 2 : select at random K points, the centroids ( not necessarily from your dataset).
    step 3 : Assign each data point to the closest centroid --> that forms K clusters.
    step 4 : compute and place the new centroid of each other
    step 5 : reassign each data point to the new closest centroid. if any reassignment 
                took place then go to step 4 otherwise go to FIN.
    
    Random Initialization Trap : the selection of centroid at the beginning of algorithm dictates the output. To correct this we need to use K-Means++ algorithm. There is no need to implement k-means ++ because its taken care of by library.

    Choosing right number of clusters : WCSS (Within Cluster Sum of Squares) Square of Distance between taken point to centroid. selecting more clusters will result in less value of WCSS. the upper limit to select the number of clusters is number of data points which would lead us to see WCSS as zero. To get optimal number of clusters we need to use Elbow method. its the value change in WCSS while increasing the number of clusters.