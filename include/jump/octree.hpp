////// TODO(jspisak): make this... interopable octree
#ifndef JUMP_OCTREE_HPP_
#define JUMP_OCTREE_HPP_

// if we have a collision we need to stop ray tracing

// basically we have a point based octree and a
// container based octree

#include <Eigen/Core>

namespace jump {

//! An axis aligned cube
class OctreeCube {
    //! The lowest (on all axes) corner
    Eigen::Matrix<double, 3, 1> ll_corner;
    //! The size of the cube
    double size;
};

//! A sphere that can be used to describe objects contained
class OctreeSphere {
    //! the center of the sphere
    Eigen::Matrix<double, 3, 1> center;
    //! The radius of the sphere
    double radius;
};

// template<typename InsertionT>
// using octree_is_spatial_type = (std::is_same<InsertionT, OctreeCube> || std::is_same<InsertionT, OctreeSphere>);

//! A leaf in an octree
template<typename ContainedT>
class OctreeLeaf {

};

//! A node within the octree (forms branches)
template<typename ContainedT>
class OctreeNode {

};

//! The root to access / modify the octree
template<typename ContainedT = int, typename InsertionT = Eigen::Matrix<double, 3, 1>>
class Octree {

    void insert(const InsertionT& i, const ContainedT& container) {

    }

}; /* class OcTree*/

} /* namespace jump */


#endif /* JUMP_OCTTREE_HPP_ */
