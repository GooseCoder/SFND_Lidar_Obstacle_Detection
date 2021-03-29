#include <vector>
#include <cmath>

#include <pcl/io/pcd_io.h>
#include <pcl/common/common.h>

// from quiz
struct Node
{
    pcl::PointXYZI point; //working with pcl points now
    int id;
    Node* left;
    Node* right;

    Node(pcl::PointXYZI arr, int setId)
    : point(arr), id(setId), left(NULL), right(NULL)
    {}
};

struct KdTree
{
    Node* root;

    KdTree() : root(NULL)
    {}

    void insertHelper(Node** node, uint depth, pcl::PointXYZI point, int id)
    {
        if(*node==NULL) {
            *node = new Node(point, id);
        }
        else
        {
            if(depth % 2 == 0)
            {
                if(point.x < ((*node)->point.x)) {
                    insertHelper(&((*node)->left), depth+1, point, id);
                }
                else {
                    insertHelper(&((*node)->right), depth+1, point, id);    
                }
            }
            else
            {
                if(point.y < ((*node)->point.y)) {
                    insertHelper(&((*node)->left), depth+1, point, id);
                }
                else {
                    insertHelper(&((*node)->right), depth+1, point, id);    
                }
            }    
        }    
    }

    void insert(pcl::PointXYZI point, int id)
    {
        insertHelper(&root, 0, point, id);
    }

    void searchHelpher(pcl::PointXYZI target, Node* node, int depth, float distanceTol, std::vector<int> &ids)
    {
        if(node!=NULL)
        {
            if( (node->point.x >= (target.x - distanceTol)) && (node->point.x <= (target.x + distanceTol)) && (node->point.y >= (target.y - distanceTol)) && (node->point.y <= (target.y + distanceTol)))
            {
                float dist = sqrt(pow((node->point.x - target.x), 2) + pow((node->point.y - target.y), 2));
                if(dist<=distanceTol) {
                    ids.push_back(node->id);
                }
            }

            if(depth % 2 == 0)
            {
                if((target.x - distanceTol) < node->point.x) {
                    searchHelpher(target, node->left, depth+1, distanceTol, ids);
                }
                if((target.x + distanceTol) > node->point.x) {
                    searchHelpher(target, node->right, depth+1, distanceTol, ids);
                }
            }
            else
            {
                if((target.y - distanceTol) < node->point.y) {
                    searchHelpher(target, node->left, depth+1, distanceTol, ids);
                }
                if((target.y + distanceTol) > node->point.y) {
                    searchHelpher(target, node->right, depth+1, distanceTol, ids);
                }
            }
        }
    }

    std::vector<int> search(pcl::PointXYZI target, float distanceTol)
    {
        std::vector<int> ids;
        searchHelpher(target, root, 0, distanceTol, ids);
        return ids;
    }
};
