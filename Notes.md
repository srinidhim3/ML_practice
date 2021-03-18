k-means clustering
    step 1 : choose the number of K clusters
    step 2 : select at random K points, the centroids ( not necessarily from your dataset).
    step 3 : Assign each data point to the closest centroid --> that forms K clusters.
    step 4 : compute and place the new centroid of each other
    step 5 : reassign each data point to the new closest centroid. if any reassignment 
                took place then go to step 4 otherwise go to FIN.
    