// PCL lib Functions for processing point clouds 

#include "processPointClouds.h"


//constructor:
template<typename PointT>
ProcessPointClouds<PointT>::ProcessPointClouds() {}


//de-constructor:
template<typename PointT>
ProcessPointClouds<PointT>::~ProcessPointClouds() {}


template<typename PointT>
void ProcessPointClouds<PointT>::numPoints(typename pcl::PointCloud<PointT>::Ptr cloud)
{
    std::cout << cloud->points.size() << std::endl;
}


template<typename PointT>
typename pcl::PointCloud<PointT>::Ptr ProcessPointClouds<PointT>::FilterCloud(typename pcl::PointCloud<PointT>::Ptr cloud, float filterRes, Eigen::Vector4f minPoint, Eigen::Vector4f maxPoint)
{

    // Time segmentation process
    auto startTime = std::chrono::steady_clock::now();

    typename pcl::PointCloud<PointT>::Ptr filteredCloud (new pcl::PointCloud<PointT>);
    pcl::VoxelGrid<PointT> voxelGridPoints;
    voxelGridPoints.setInputCloud(cloud);
    voxelGridPoints.setLeafSize(filterRes, filterRes, filterRes);
    voxelGridPoints.filter(*filteredCloud);

    //filter out a region of interest
    typename pcl::PointCloud<PointT>::Ptr cloudRegion(new pcl::PointCloud<PointT>);
    pcl::CropBox<PointT> filter;
    filter.setInputCloud(filteredCloud);
    filter.setMin(minPoint);
    filter.setMax(maxPoint);
    filter.filter(*cloudRegion);

    //remove the removeRoof points
    pcl::CropBox<PointT> removeRoof(true);
    removeRoof.setMin(Eigen::Vector4f(-1.5, -1.7,-1,1));
    removeRoof.setMax(Eigen::Vector4f(2.6, 1.7, -.4, 1));
    removeRoof.setInputCloud(cloudRegion);
    std::vector<int> indices;
    removeRoof.filter(indices);

    pcl::PointIndices::Ptr inlinePoints (new pcl::PointIndices);
    for (int point : indices)
      inlinePoints->indices.push_back(point);
    pcl::ExtractIndices<PointT> extract;
    extract.setInputCloud(cloudRegion);
    extract.setIndices(inlinePoints);
    extract.setNegative(true);
    extract.filter(*cloudRegion);

    auto endTime = std::chrono::steady_clock::now();
    auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    std::cout << "filtering took " << elapsedTime.count() << " milliseconds" << std::endl;

    return cloudRegion;

}

template<typename PointT>
std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr> ProcessPointClouds<PointT>::SeparateClouds(pcl::PointIndices::Ptr inlinePoints, typename pcl::PointCloud<PointT>::Ptr cloud) 
{
    typename pcl::PointCloud<PointT>::Ptr obstCloud (new pcl::PointCloud<PointT>());
    typename pcl::PointCloud<PointT>::Ptr planeCloud (new pcl::PointCloud<PointT>());
    pcl::ExtractIndices<PointT> extract;
    extract.setInputCloud(cloud);
    extract.setIndices(inlinePoints);
    extract.setNegative(false);
    extract.filter(*planeCloud);

    extract.setNegative(true);
    extract.filter(*obstCloud);

    std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr> segResult(obstCloud, planeCloud);
    return segResult;
}

template<typename PointT>
std::unordered_set<int> ProcessPointClouds<PointT>::myRansac(typename pcl::PointCloud<PointT>::Ptr cloud, int maxIterations, float distanceTol)
{
    std::unordered_set<int> iinliers;
    std::unordered_set<int> inliersResult;
    srand(time(NULL));

    int numcloud = cloud->points.size();
    for(int i=0; i<maxIterations; i++){
        iinliers.clear();
        for (int j=0; j < 3; j++) {
            iinliers.insert(rand() % (numcloud));
        }

        // sample 3 points
        PointT p1=cloud->points[(rand() % numcloud)];
        PointT p2=cloud->points[(rand() % numcloud)];
        PointT p3=cloud->points[(rand() % numcloud)];

        // fit plane
        float A = (p2.y - p1.y)*(p3.z - p1.z) - (p2.z - p1.z)*(p3.y - p1.y);
        float B = (p2.z - p1.z)*(p3.x - p1.x) - (p2.x - p1.x)*(p3.z - p1.z);
        float C = (p2.x - p1.x)*(p3.y - p1.y) - (p2.y - p1.y)*(p3.x - p1.x);
        float D = -(A*p1.x + B*p1.y + C*p1.z);
        // iterate through all points in the pointcloud
        for (int index=0; index < numcloud; index++) {
            if (iinliers.count(index)>0) {
                //skip if in iinliers already
                continue;
            }

            // for each point, do distance test
            PointT point = cloud->points[index];
            float ds = fabs(A*point.x + B*point.y + C*point.z + D) / sqrt(A*A + B*B + C*C);
            if (ds <= distanceTol) {
                iinliers.insert(index);
            }
        }

        if (iinliers.size() > inliersResult.size()) {
            inliersResult = iinliers;
        }
    }
    return inliersResult;
}


template<typename PointT>
std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr> ProcessPointClouds<PointT>::mySegmentPlane(typename pcl::PointCloud<PointT>::Ptr cloud, int maxIterations, float distanceThreshold)
{
    // Time segmentation process
    auto startTime = std::chrono::steady_clock::now();

    pcl::PointIndices::Ptr inlinePoints (new pcl::PointIndices());

    std::unordered_set<int> inliersResult;
    inliersResult = myRansac(cloud, maxIterations, distanceThreshold);

    for(int ind: inliersResult) {
        inlinePoints->indices.push_back(ind);  
    }
    
    if (inlinePoints->indices.size () == 0)
    {
      std::cout << "Could not estimate a planar model for the given dataset." << std::endl;
    }

    auto endTime = std::chrono::steady_clock::now();
    auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    std::cout << "plane segmentation took " << elapsedTime.count() << " milliseconds" << std::endl;

    std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr> segResult = SeparateClouds(inlinePoints,cloud);
    return segResult;
}

template<typename PointT>
std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr> ProcessPointClouds<PointT>::SegmentPlane(typename pcl::PointCloud<PointT>::Ptr cloud, int maxIterations, float distanceThreshold)
{
    // Time segmentation process
    auto startTime = std::chrono::steady_clock::now();
    pcl::SACSegmentation<PointT> seg;
    pcl::PointIndices::Ptr inlinePoints (new pcl::PointIndices());
    pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients());

    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setMaxIterations(maxIterations);
    seg.setDistanceThreshold(distanceThreshold);

    seg.setInputCloud(cloud);
    seg.segment(*inlinePoints, *coefficients);
    if (inlinePoints->indices.size() == 0) {
      std::cout << "Could not estimate a planar model for the given dataset." << std::endl;
    }

    auto endTime = std::chrono::steady_clock::now();
    auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    std::cout << "plane segmentation took " << elapsedTime.count() << " milliseconds" << std::endl;

    std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr> segResult = SeparateClouds(inlinePoints,cloud);
    return segResult;
}

template<typename PointT>
std::vector<typename pcl::PointCloud<PointT>::Ptr> ProcessPointClouds<PointT>::Clustering(typename pcl::PointCloud<PointT>::Ptr cloud, float clusterTolerance, int minSize, int maxSize)
{
    auto startTime = std::chrono::steady_clock::now();

    std::vector<typename pcl::PointCloud<PointT>::Ptr> clusters;

    typename pcl::search::KdTree<PointT>::Ptr tree (new pcl::search::KdTree<PointT>);
    tree->setInputCloud(cloud);

    std::vector<pcl::PointIndices> clusterIndices;
    pcl::EuclideanClusterExtraction<PointT> ec;
    ec.setClusterTolerance(clusterTolerance);
    ec.setMinClusterSize(minSize);
    ec.setMaxClusterSize(maxSize);
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloud);
    ec.extract(clusterIndices);

    for (std::vector<pcl::PointIndices>::const_iterator it = clusterIndices.begin(); it != clusterIndices.end(); ++it) {
      typename pcl::PointCloud<PointT>::Ptr cloudCluster (new pcl::PointCloud<PointT>);
      for (std::vector<int>::const_iterator pit = it->indices.begin(); pit != it->indices.end(); ++pit) {
        cloudCluster->points.push_back(cloud->points[*pit]);
      }
      cloudCluster->width = cloudCluster->points.size();
      cloudCluster->height = 1;
      cloudCluster->is_dense = true;
      clusters.push_back(cloudCluster);
    }

    auto endTime = std::chrono::steady_clock::now();
    auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    std::cout << "clustering took " << elapsedTime.count() << " milliseconds and found " << clusters.size() << " clusters" << std::endl;

    return clusters;
}

template<typename PointT>           
void ProcessPointClouds<PointT>::clusterHelper(int i, typename pcl::PointCloud<PointT>::Ptr cloud, KdTree* tree, float distanceTol, std::vector<bool> &processed, std::vector<int> &cluster)
{
    // from quiz
    processed[i] = true;
    cluster.push_back(i);
    std::vector<int> nearest = tree->search(cloud->points[i], distanceTol);
    for(int k : nearest)
    {
        if(!processed[k]) {
            clusterHelper(k, cloud, tree, distanceTol, processed, cluster);
        }
    }
}

template<typename PointT>
std::vector<typename pcl::PointCloud<PointT>::Ptr> ProcessPointClouds<PointT>::myClustering(typename pcl::PointCloud<PointT>::Ptr cloud, float clusterTolerance, int minSize, int maxSize)
{
    auto startTime = std::chrono::steady_clock::now();

    std::vector<typename pcl::PointCloud<PointT>::Ptr> clusters;
    std::vector<std::vector<int>> clusters_idx;
    std::vector<bool> processed (cloud->points.size(), false);

    // from quiz
    KdTree *tree = new KdTree;
    for (int i=0; i<cloud->points.size(); ++i) {
        tree->insert(cloud->points[i], i);
    }

    for (int i=0; i < cloud->points.size(); ++i) {
        if (processed[i]) {
            continue;
        }
        std::vector<int> cluster_idx;
        clusterHelper(i, cloud, tree, clusterTolerance, processed, cluster_idx);
        clusters_idx.push_back(cluster_idx);
    }

    for (std::vector<int> idx : clusters_idx) {
        if(idx.size() < minSize || idx.size() > maxSize) {
            continue;
        }
        
        typename pcl::PointCloud<PointT>::Ptr cloudCluster(new pcl::PointCloud<PointT>());
        for (int ind : idx) {
            cloudCluster->points.push_back(cloud->points[ind]);
        }
        cloudCluster->width = cloudCluster->points.size();
        cloudCluster->height = 1;
        cloudCluster->is_dense = true;

        clusters.push_back(cloudCluster);
    }

    auto endTime = std::chrono::steady_clock::now();
    auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    std::cout << "clustering took " << elapsedTime.count() << " milliseconds and found " << clusters.size() << " clusters" << std::endl;

    return clusters;
}

template<typename PointT>
Box ProcessPointClouds<PointT>::BoundingBox(typename pcl::PointCloud<PointT>::Ptr cluster)
{
    // Find bounding box for one of the clusters
    PointT minPoint, maxPoint;
    pcl::getMinMax3D(*cluster, minPoint, maxPoint);

    Box box;
    box.x_min = minPoint.x;
    box.y_min = minPoint.y;
    box.z_min = minPoint.z;
    box.x_max = maxPoint.x;
    box.y_max = maxPoint.y;
    box.z_max = maxPoint.z;

    return box;
}


template<typename PointT>
void ProcessPointClouds<PointT>::savePcd(typename pcl::PointCloud<PointT>::Ptr cloud, std::string file)
{
    pcl::io::savePCDFileASCII (file, *cloud);
    std::cerr << "Saved " << cloud->points.size () << " data points to "+file << std::endl;
}


template<typename PointT>
typename pcl::PointCloud<PointT>::Ptr ProcessPointClouds<PointT>::loadPcd(std::string file)
{

    typename pcl::PointCloud<PointT>::Ptr cloud (new pcl::PointCloud<PointT>);

    if (pcl::io::loadPCDFile<PointT> (file, *cloud) == -1) //* load the file
    {
        PCL_ERROR ("Couldn't read file \n");
    }
    std::cerr << "Loaded " << cloud->points.size () << " data points from "+file << std::endl;

    return cloud;
}


template<typename PointT>
std::vector<boost::filesystem::path> ProcessPointClouds<PointT>::streamPcd(std::string dataPath)
{

    std::vector<boost::filesystem::path> paths(boost::filesystem::directory_iterator{dataPath}, boost::filesystem::directory_iterator{});

    // sort files in accending order so playback is chronological
    sort(paths.begin(), paths.end());

    return paths;

}
