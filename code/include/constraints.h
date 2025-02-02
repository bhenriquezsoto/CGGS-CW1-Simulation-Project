#ifndef CONSTRAINTS_HEADER_FILE
#define CONSTRAINTS_HEADER_FILE

using namespace Eigen;
using namespace std;

typedef enum ConstraintType{DISTANCE, COLLISION} ConstraintType;   //You can expand it for more constraints
typedef enum ConstraintEqualityType{EQUALITY, INEQUALITY} ConstraintEqualityType;

//there is such constraints per two variables that are equal. That is, for every attached vertex there are three such constraints for (x,y,z);
class Constraint{
public:
  
  int m1, m2;                     //Two participating meshes (can be the same)  - auxiliary data for users (constraint class shouldn't use that)
  int v1, v2;                     //Two vertices from the respective meshes - auxiliary data for users (constraint class shouldn't use that)
  double invMass1, invMass2;       //inverse masses of two bodies
  double refValue;                //Reference values to use in the constraint, when needed (like distance)
  bool isUpper;                   //in case this is an inequality constraints, whether it's an upper or a lower bound
  RowVector3d refVector;             //Reference vector when needed (like vector)
  double CRCoeff;                 //velocity bias
  ConstraintType constraintType;  //The type of the constraint, and will affect the value and the gradient. This SHOULD NOT change after initialization!
  ConstraintEqualityType constraintEqualityType;  //whether the constraint is an equality or an inequality
  
  Constraint(const ConstraintType _constraintType, const ConstraintEqualityType _constraintEqualityType, const bool _isUpper, const int& _m1, const int& _v1, const int& _m2, const int& _v2, const double& _invMass1, const double& _invMass2, const RowVector3d& _refVector, const double& _refValue, const double& _CRCoeff):constraintType(_constraintType), constraintEqualityType(_constraintEqualityType), isUpper(_isUpper), m1(_m1), v1(_v1), m2(_m2), v2(_v2), invMass1(_invMass1), invMass2(_invMass2),  refValue(_refValue), CRCoeff(_CRCoeff){
    refVector=_refVector;
  }
  
  ~Constraint(){}
  
  
  
  //computes the impulse needed for all particles to resolve the velocity constraint, and corrects the velocities accordingly.
  //The velocities are a vector (vCOM1, w1, vCOM2, w2) in both input and output.
  //returns true if constraint was already valid with "currVelocities", and false otherwise (false means there was a correction done)
  bool resolve_velocity_constraint(const MatrixXd& currCOMPositions, const MatrixXd& currVertexPositions, const MatrixXd& currCOMVelocities, const MatrixXd& currAngVelocities, const Matrix3d& invInertiaTensor1, const Matrix3d& invInertiaTensor2, MatrixXd& correctedCOMVelocities, MatrixXd& correctedAngVelocities, double tolerance){
    
    
    /***************************TODO: implement this function**********************/
    
    //stub implementation
    correctedCOMVelocities=currCOMVelocities;
    correctedAngVelocities=currAngVelocities;
    return true;
    
  }
  
  //projects the position unto the constraint
  //returns true if constraint was already good
  bool resolve_position_constraint(const MatrixXd& currCOMPositions, const MatrixXd& currConstPositions, MatrixXd& correctedCOMPositions, double tolerance){
    
    /***************************TODO: implement this function**********************/
    
    //stub implementation
    correctedCOMPositions=currCOMPositions;
    return true;
  }
  
};



#endif /* constraints_h */
