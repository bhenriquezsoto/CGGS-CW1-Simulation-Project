#ifndef AABBTREE_HEADER_FILE
#define AABBTREE_HEADER_FILE

#include <vector>
#include <Eigen/Core>
#include "mesh.h"

using namespace Eigen;
using namespace std;

const double AABB_MARGIN = 0.1;

struct AABB {
    RowVector3d min;
    RowVector3d max;

    AABB() {
        min = RowVector3d::Constant(std::numeric_limits<double>::max());
        max = RowVector3d::Constant(std::numeric_limits<double>::lowest());
    }

    AABB(const RowVector3d& min_, const RowVector3d& max_) : min(min_), max(max_) {}

    // Check if this AABB intersects with another
    bool intersects(const AABB& other) const {
        return !(max(0) < other.min(0) || min(0) > other.max(0) ||
                max(1) < other.min(1) || min(1) > other.max(1) ||
                max(2) < other.min(2) || min(2) > other.max(2));
    }

    // Compute union of two AABBs
    AABB unite(const AABB& other) const {
        return AABB(
            min.cwiseMin(other.min),
            max.cwiseMax(other.max)
        );
    }
};

class AABBNode {
public:
    AABB box;
    AABBNode* left;
    AABBNode* right;
    int meshIndex;  // Index of the mesh this node contains (-1 if internal node)

    AABBNode() : left(nullptr), right(nullptr), meshIndex(-1) {}
};

class AABBTree {
private:
    AABBNode* root;
    vector<Mesh*>& meshes;

    // Find closest pair of nodes
    pair<AABBNode*, AABBNode*> findClosestNodes(vector<AABBNode*>& nodes) {
        double minCost = std::numeric_limits<double>::max();
        pair<AABBNode*, AABBNode*> closestPair;
        
        for(size_t i = 0; i < nodes.size(); i++) {
            for(size_t j = i + 1; j < nodes.size(); j++) {
                AABB combined = nodes[i]->box.unite(nodes[j]->box);
                // Cost could be volume or surface area of combined box
                double cost = (combined.max - combined.min).norm();
                
                if(cost < minCost) {
                    minCost = cost;
                    closestPair = {nodes[i], nodes[j]};
                }
            }
        }
        return closestPair;
    }

    // Build tree bottom-up (from the leaves to the root)
    void buildBottomUp() {
        // Create leaf nodes for all meshes
        vector<AABBNode*> nodes;
        for(size_t i = 0; i < meshes.size(); i++) {
            AABBNode* leaf = new AABBNode();
            leaf->meshIndex = i;
            leaf->box = computeMeshAABB(*meshes[i]);
            nodes.push_back(leaf);
        }

        // Iteratively combine closest pairs
        while(nodes.size() > 1) {
            // Find closest pair of nodes
            auto [node1, node2] = findClosestNodes(nodes);
            
            // Create parent node
            AABBNode* parent = new AABBNode();
            parent->left = node1;
            parent->right = node2;
            parent->box = node1->box.unite(node2->box);
            parent->meshIndex = -1;  // Internal node
            
            // Remove children from nodes list and add parent
            nodes.erase(remove(nodes.begin(), nodes.end(), node1), nodes.end());
            nodes.erase(remove(nodes.begin(), nodes.end(), node2), nodes.end());
            nodes.push_back(parent);
        }
        
        root = nodes[0];  // Last remaining node is root
    }

    void deleteTree(AABBNode* node) {
        if (node) {
            deleteTree(node->left);
            deleteTree(node->right);
            delete node;
        }
    }

    // Helper function to find potential collisions
    void findCollisionsRecursive(AABBNode* node1, AABBNode* node2, vector<pair<int, int>>& collisionPairs) {
        if (!node1 || !node2 || !node1->box.intersects(node2->box)) return;

        // If both are leaves, add the pair to collision candidates
        if (node1->meshIndex != -1 && node2->meshIndex != -1) {
            if (node1->meshIndex < node2->meshIndex) {
                collisionPairs.push_back({node1->meshIndex, node2->meshIndex});
            }
            return;
        }

        // Recursively check children
        if (node1->meshIndex != -1) {
            findCollisionsRecursive(node1, node2->left, collisionPairs);
            findCollisionsRecursive(node1, node2->right, collisionPairs);
        } else if (node2->meshIndex != -1) {
            findCollisionsRecursive(node1->left, node2, collisionPairs);
            findCollisionsRecursive(node1->right, node2, collisionPairs);
        } else {
            findCollisionsRecursive(node1->left, node2->left, collisionPairs);
            findCollisionsRecursive(node1->left, node2->right, collisionPairs);
            findCollisionsRecursive(node1->right, node2->left, collisionPairs);
            findCollisionsRecursive(node1->right, node2->right, collisionPairs);
        }
    }

public:
    AABBTree(vector<Mesh*>& meshes_) : meshes(meshes_), root(nullptr) {
        buildBottomUp();
    }

    ~AABBTree() {
        deleteTree(root);
    }

    // Compute AABB for a mesh with margin
    AABB computeMeshAABB(const Mesh& mesh) const {
        AABB box;
        for (int i = 0; i < mesh.currV.rows(); i++) {
            box.min = box.min.cwiseMin(mesh.currV.row(i));
            box.max = box.max.cwiseMax(mesh.currV.row(i));
        }
        
        // Add margin to the bounding box
        RowVector3d margin = RowVector3d::Constant(AABB_MARGIN);
        box.min -= margin;
        box.max += margin;
        
        return box;
    }

    // Get AABB for a specific mesh
    AABB getMeshAABB(int meshIndex) const {
        if (meshIndex >= 0 && meshIndex < meshes.size()) {
            return computeMeshAABB(*meshes[meshIndex]);
        }
        return AABB();
    }

    // Rebuild the entire tree
    void rebuild() {
        deleteTree(root);
        if (meshes.empty()) {
            root = nullptr;
            return;
        }
        buildBottomUp();
    }

    void update() {
        if (!root) return;
        // In this case, we are rebuilding the tree but a
        // more sophisticated version could update boxes in place
        rebuild();
    }

    // Find all potential collision pairs
    vector<pair<int, int>> findPotentialCollisions() {
        vector<pair<int, int>> collisionPairs;
        findCollisionsRecursive(root, root, collisionPairs);
        return collisionPairs;
    }
};

#endif
